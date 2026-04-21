// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "LatencyModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "modulo-scheduling-latency"

namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {

llvm::StringRef getPipelineName(HWPipeline pipeline) {
  switch (pipeline) {
  case HWPipeline::MEM:
    return "MEM";
  case HWPipeline::TC:
    return "TC";
  case HWPipeline::CUDA:
    return "CUDA";
  case HWPipeline::SFU:
    return "SFU";
  case HWPipeline::NONE:
    return "NONE";
  }
  llvm_unreachable("unknown pipeline");
}

// Estimate total elements in the result tensor of an op.
int64_t LatencyModel::getTensorElements(Operation *op) const {
  if (op->getNumResults() == 0)
    return 0;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return 0;
  int64_t elements = 1;
  for (auto dim : resultType.getShape())
    elements *= dim;
  return elements;
}

// TMA load latencies from B200 microbenchmarks (cycles).
// Key = total bytes, value = pipeline occupancy cycles.
// Entries from NVIDIA_B200_latency_table.json.
struct TMALatencyEntry {
  int64_t bytes;
  int cycles;
};
static constexpr TMALatencyEntry kTMALoadTable[] = {
    {128 * 64 * 2, 518},  // 128x64 or 64x128 bf16/fp16 = 16KB
    {128 * 128 * 2, 654}, // 128x128 bf16/fp16 = 32KB
    {256 * 64 * 2, 653},  // 256x64 bf16 = 32KB
    {256 * 128 * 2, 918}, // 256x128 bf16 = 64KB
};

// Async overhead: additional cycles for data to travel through the memory
// hierarchy (L2/DRAM) and arrive in SMEM. On top of pipeline occupancy.
constexpr int kTMAAsyncOverhead = 700;

/// Look up TMA load occupancy by total bytes. Table lookup first, then
/// linear interpolation from 128x64 baseline as fallback.
static int lookupTMALoadOccupancy(int64_t totalBytes) {
  for (const auto &entry : kTMALoadTable) {
    if (entry.bytes == totalBytes)
      return entry.cycles;
  }
  // Fallback: linear interpolation from 128x64 baseline.
  constexpr int64_t kBaseBytes = 128 * 64 * 2;
  constexpr int kBaseCycles = 518;
  return static_cast<int>(kBaseCycles * static_cast<double>(totalBytes) /
                          kBaseBytes);
}

int LatencyModel::getTMALoadLatency(Operation *op) const {
  if (op->getNumResults() == 0)
    return lookupTMALoadOccupancy(128 * 64 * 2); // default: 128x64
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return lookupTMALoadOccupancy(128 * 64 * 2);

  int64_t elements = 1;
  for (auto dim : resultType.getShape())
    elements *= dim;
  int64_t bytesPerElement = resultType.getElementTypeBitWidth() / 8;
  return lookupTMALoadOccupancy(elements * bytesPerElement);
}

int LatencyModel::getTMAStoreLatency(Operation *op) const {
  // TMA stores have similar latency profile to loads
  return getTMALoadLatency(op);
}

// MMA latencies from design doc microbenchmarks (Blackwell tcgen05.mma).
// Scales with the product M*N*K.
constexpr int kMMALatency128x128x128 = 900;
constexpr int kMMALatency128x128x64 = 559;

int LatencyModel::getMMALatency(Operation *op) const {
  if (op->getNumResults() == 0)
    return kMMALatency128x128x128; // conservative default
  // Try to extract the MMA shape from the MMAv5 interface
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto aType = dyn_cast<RankedTensorType>(mma->getOperand(0).getType());
    auto bType = dyn_cast<RankedTensorType>(mma->getOperand(1).getType());
    if (aType && bType) {
      auto aShape = aType.getShape(); // [M, K]
      int64_t K = aShape.size() >= 2 ? aShape[1] : 64;
      // Use K to select between known latencies
      if (K >= 128)
        return kMMALatency128x128x128;
      return kMMALatency128x128x64;
    }
  }
  return kMMALatency128x128x128; // conservative default
}

int LatencyModel::getCUDALatency(Operation *op) const {
  // Ops that don't produce tensor results but have real latency.
  // Check these before the scalar early-return.
  if (isa<ttng::WaitBarrierOp>(op))
    return 30;
  if (isa<ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(op))
    return 20;
  if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
    return 105;
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return 105;

  int64_t elements = getTensorElements(op);
  if (elements == 0)
    return 0; // scalar

  // Reductions: differentiate by reduction kind.
  if (auto reduceOp = dyn_cast<tt::ReduceOp>(op)) {
    // RowMax ~336 cycles, RowSum ~508 cycles for 128-wide (from microbench).
    // Heuristic: check if the reduction body contains an AddF (sum) or MaxF.
    bool isSum = false;
    reduceOp.getBody()->walk([&](Operation *inner) {
      if (isa<arith::AddFOp>(inner))
        isSum = true;
    });
    return isSum ? 508 : 336; // RowSum vs RowMax
  }

  // Type conversions (truncf, extf): ~105 cycles for 128x128.
  if (isa<arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp, arith::SIToFPOp,
          tt::FpToFpOp, tt::BitcastOp>(op))
    return 105;

  // Multiply (Acc x Alpha): ~105 cycles for 128x128.
  if (isa<arith::MulFOp>(op))
    return 105;

  // TMEM load/store, SMEM load/store, layout conversions: ~105 cycles.
  if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, triton::gpu::LocalLoadOp,
          triton::gpu::LocalStoreOp, triton::gpu::ConvertLayoutOp>(op))
    return 105;

  // Integer type conversions: ~105 cycles (same as float conversions).
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp>(
          op))
    return 105;

  // Integer arithmetic, comparisons, selects, other elementwise: ~130 cycles.
  return 130;
}

int LatencyModel::getSFULatency(Operation *op) const {
  int64_t elements = getTensorElements(op);
  if (elements == 0)
    return 43; // scalar exp2 (Alpha = Exp2(scalar))
  return 662;  // elementwise exp2 for 128x128
}

HWPipeline LatencyModel::classifyPipeline(Operation *op) const {
  // MEM: TMA loads, regular loads, and stores
  if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
    return HWPipeline::MEM;
  // MEM: Lowered TMA loads (TLX kernels use async_tma_copy instead of
  // descriptor_load)
  if (isa<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return HWPipeline::MEM;
  if (isa<tt::LoadOp>(op)) {
    // Regular tt.load (before TMA lowering) — classify as MEM if tensor
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::MEM;
  }
  if (isa<tt::DescriptorStoreOp>(op))
    return HWPipeline::MEM;
  // MEM: Lowered TMA stores (TLX path)
  if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return HWPipeline::MEM;

  // TC: Tensor Core MMA operations
  if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp>(op))
    return HWPipeline::TC;
  if (isa<ttng::WarpGroupDotOp>(op))
    return HWPipeline::TC;
  // TC: tt.dot (before lowering to TCGen5MMAOp / WarpGroupDotOp)
  if (isa<tt::DotOp>(op))
    return HWPipeline::TC;

  // CUDA: TMEM load/store (data movement between registers and TMEM)
  if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
    return HWPipeline::CUDA;

  // CUDA: SMEM load/store (data movement between registers and SMEM)
  if (isa<triton::gpu::LocalLoadOp, triton::gpu::LocalStoreOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Layout conversions on tensors (may involve SMEM round-trips)
  if (isa<triton::gpu::ConvertLayoutOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Barrier operations (synchronization between warp groups).
  // These carry timing dependencies between producers and consumers
  // in warp-specialized kernels.
  if (isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp, ttng::BarrierExpectOp>(
          op))
    return HWPipeline::CUDA;

  // MEM: Regular tensor stores to global memory
  if (isa<tt::StoreOp>(op)) {
    if (op->getNumOperands() > 1) {
      auto valOperand = op->getOperand(1);
      if (isa<RankedTensorType>(valOperand.getType()))
        return HWPipeline::MEM;
    }
  }

  // SFU: Transcendental math operations on tensors
  if (isa<math::Exp2Op, math::ExpOp, math::Log2Op, math::LogOp, math::SqrtOp,
          math::RsqrtOp, math::TanhOp>(op)) {
    // Only classify as SFU if operating on tensors
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::SFU;
    return HWPipeline::NONE; // scalar math is free
  }

  // CUDA: Reductions
  if (isa<tt::ReduceOp>(op))
    return HWPipeline::CUDA;

  // CUDA: Tensor arithmetic (elementwise operations on tensors)
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::MaximumFOp, arith::MinimumFOp, arith::NegFOp, arith::CmpFOp,
          arith::CmpIOp, arith::SelectOp>(op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::CUDA;
  }

  // CUDA: Integer tensor arithmetic (index computation, masking)
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
          arith::DivSIOp, arith::RemUIOp, arith::RemSIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRUIOp,
          arith::ShRSIOp>(op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::CUDA;
  }

  // CUDA: Integer type conversions on tensors
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp>(
          op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::CUDA;
  }

  // CUDA: Float type conversions on tensors
  if (isa<arith::TruncFOp, arith::ExtFOp, arith::FPToSIOp, arith::SIToFPOp,
          tt::FpToFpOp, tt::BitcastOp>(op)) {
    if (op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType()))
      return HWPipeline::CUDA;
  }

  // MEM: local_alloc fed by a MEM load represents the async data arrival.
  // It stays at the same stage as the load (edge uses selfLatency), but
  // carries the async overhead latency to its consumers (MMA).
  if (isa<triton::gpu::LocalAllocOp>(op)) {
    // Check if operand comes from a load
    if (op->getNumOperands() > 0) {
      auto *srcOp = op->getOperand(0).getDefiningOp();
      if (srcOp && (isa<tt::LoadOp, tt::DescriptorLoadOp>(srcOp)))
        return HWPipeline::MEM;
    }
  }

  // NONE: Scalar ops, index arithmetic, control flow, barriers, etc.
  return HWPipeline::NONE;
}

OpLatencyInfo LatencyModel::getLatency(Operation *op) const {
  auto pipeline = classifyPipeline(op);

  int latency = 0;
  int selfLatency = 0;
  switch (pipeline) {
  case HWPipeline::MEM: {
    // For async MEM ops, selfLatency (pipeline occupancy) and latency
    // (time until data available for consumers) are different.
    // selfLatency = how long the MEM pipeline is busy dispatching.
    // latency = selfLatency + async overhead (DRAM round-trip).
    int occupancy;
    if (isa<tt::DescriptorStoreOp>(op))
      occupancy = getTMAStoreLatency(op);
    else if (isa<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      // Lowered TMA store — use same logic as descriptor_store.
      occupancy = lookupTMALoadOccupancy(128 * 64 * 2);
    } else if (isa<triton::gpu::LocalAllocOp>(op)) {
      // local_alloc fed by a load: represents async data arrival.
      // selfLatency = 0 (no pipeline occupancy, it's a bookkeeping op).
      // latency = async overhead (DRAM round-trip time).
      selfLatency = 0;
      latency = kTMAAsyncOverhead;
      break;
    } else if (auto tmaCopy = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      // Lowered TMA load (TLX path). Get size from the SMEM result type.
      auto resultMemDesc =
          dyn_cast<triton::gpu::MemDescType>(tmaCopy.getResult().getType());
      if (resultMemDesc) {
        int64_t elements = 1;
        for (auto dim : resultMemDesc.getShape())
          elements *= dim;
        int64_t bytesPerElement =
            resultMemDesc.getElementType().getIntOrFloatBitWidth() / 8;
        occupancy = lookupTMALoadOccupancy(elements * bytesPerElement);
      } else {
        occupancy = lookupTMALoadOccupancy(128 * 64 * 2);
      }
    } else
      occupancy = getTMALoadLatency(op);
    // selfLatency = 1: GPU TMA unit is deeply pipelined and can accept
    // new requests every cycle. The occupancy value reflects data transfer
    // time, not issue blocking. Using occupancy as selfLatency inflates
    // ResMII and causes modulo scheduling to fail on kernels with many
    // loads (e.g., FA backward with 6 MEM ops would need ResMII=3400+).
    selfLatency = 1;
    latency = occupancy + kTMAAsyncOverhead;
    break;
  }
  case HWPipeline::TC:
    latency = getMMALatency(op);
    // selfLatency = 1: GPU tensor core pipeline is deeply pipelined —
    // a new MMA can be issued every ~1-32 cycles while the previous one
    // is still computing. Using latency (900 cycles) as selfLatency
    // inflates ResMII to 4500 for 5 MMAs, causing SMS to fail.
    selfLatency = 1;
    break;
  case HWPipeline::CUDA:
    latency = getCUDALatency(op);
    // selfLatency = 1: CUDA ALUs are wide vector units that can accept
    // new instructions every cycle. The latency value reflects execution
    // time, not issue blocking.
    selfLatency = 1;
    break;
  case HWPipeline::SFU:
    latency = getSFULatency(op);
    // selfLatency = 1: SFU is pipelined, accepts new instructions quickly.
    selfLatency = 1;
    break;
  case HWPipeline::NONE:
    latency = 0;
    selfLatency = 0;
    break;
  }

  return OpLatencyInfo{pipeline, latency, selfLatency};
}

} // namespace mlir::triton::gpu
