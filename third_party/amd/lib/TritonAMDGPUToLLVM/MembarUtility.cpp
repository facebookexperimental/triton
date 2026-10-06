#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {
// Returns true for a producer-to-consumer dependency ordered by AsyncWait.
// AsyncWait does not release the consumed LDS slice for a later async write.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       bool op1IsRead, bool op2IsRead,
                                       Allocation *allocation) {
  auto isAsyncLDSWrite = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadSyncedViaAsyncWait = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };
  auto getMemdescValue = [](Operation *op) -> Value {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<triton::amdgpu::BufferLoadToLocalOp>(
            [](auto op) { return op.getDest(); })
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp>(
            [](auto op) { return op.getResult(); })
        .Case<triton::gpu::LocalLoadOp>([](auto op) { return op.getSrc(); })
        .Default([](Operation *) { return Value(); });
  };

  // Only filter a RAW dependency from a prior async LDS write to its local
  // consumer. In particular, never filter the opposite LocalLoad-to-prefetch
  // WAR dependency: a wait says nothing about consumer completion.
  if (op1IsRead || !op2IsRead || !isAsyncLDSWrite(op1) ||
      !isLocalLoadSyncedViaAsyncWait(op2)) {
    return false;
  }

  Value op1Memdesc = getMemdescValue(op1);
  Value op2Memdesc = getMemdescValue(op2);
  if (!op1Memdesc || !op2Memdesc)
    return false;
  auto op1BufferIds = allocation->getAllBufferIdsWithAliases(op1Memdesc);
  auto op2BufferIds = allocation->getAllBufferIdsWithAliases(op2Memdesc);

  // Check if operations access the same buffer
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });

  if (!sameBuffer)
    return false;

  return true;
}

// A deliberately narrow reuse rule. Equal concrete tensor types select
// equal shared swizzles, and equal allocation intervals select the same base.
// Warp-local, injective layouts therefore give both conversions the same
// disjoint per-wave byte partitions. Membar pairs this proof with a leading
// wave fence only when allocated scratch has an unsynchronized read-to-write
// dependency, without rendezvousing waves.
bool filterWarpLocalScratchReuse(Operation *op1, Operation *op2,
                                 Allocation *allocation) {
  auto first = dyn_cast<triton::gpu::ConvertLayoutOp>(op1);
  auto second = dyn_cast<triton::gpu::ConvertLayoutOp>(op2);
  if (!first || !second || op1 == op2 || op1->getBlock() != op2->getBlock() ||
      !op1->isBeforeInBlock(op2))
    return false;
  auto srcTy = cast<RankedTensorType>(first.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(first.getType());
  if (!srcTy.getEncoding() || !dstTy.getEncoding() ||
      srcTy != second.getSrc().getType() || dstTy != second.getType())
    return false;

  auto firstId = allocation->getBufferId(op1);
  auto secondId = allocation->getBufferId(op2);
  if (firstId == Allocation::InvalidBufferId ||
      secondId == Allocation::InvalidBufferId)
    return false;
  auto interval = allocation->getAllocatedInterval(firstId);
  if (interval != allocation->getAllocatedInterval(secondId))
    return false;

  // Restrict this first rule to one complete, non-broadcast scratch image.
  // Partial/repeated scratch images need their physical wave partition proved
  // separately; different bases cannot be accepted on logical layout alone.
  auto elemTy = srcTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return false;
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  if ((bitwidth != 8 && bitwidth != 16 && bitwidth != 32 && bitwidth != 64) ||
      interval.size() != srcTy.getNumElements() * (bitwidth / 8))
    return false;
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  if (srcLayout.isModular() || dstLayout.isModular() ||
      !srcLayout.isInjective() || !dstLayout.isInjective())
    return false;
  // Equal types can select different lowerings through forceWarpShuffle.
  // Both operations must actually use the shared-memory scratch image.
  if (!mlir::cvtNeedsSharedMemory(first) || !mlir::cvtNeedsSharedMemory(second))
    return false;
  auto warp = StringAttr::get(op1->getContext(), "warp");
  return mlir::isCvtDimSync(srcLayout, dstLayout, warp);
}

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, op1IsRead, op2IsRead,
                                            allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}

MembarScratchSync getWarpLocalScratchSync() {
  return {filterWarpLocalScratchReuse, [](Operation *op, OpBuilder &builder) {
            // Repeated Membar runs retain the real fence rather than trusting
            // an input attribute to stand for synchronization.
            auto previous =
                dyn_cast_or_null<LLVM::CallIntrinsicOp>(op->getPrevNode());
            if (previous && previous.getIntrin() == "llvm.amdgcn.wave.barrier")
              return;
            LLVM::createLLVMIntrinsicCallOp(builder, op->getLoc(),
                                            "llvm.amdgcn.wave.barrier", {}, {});
          }};
}

namespace {
// External model that stamps the marker interface onto an upstream ROCDL op we
// do not own. The interface has no methods, so the model body is empty.
template <typename OpT>
struct SchedulingBarrierModel
    : public ::mlir::triton::gpu::SchedulingBarrierOpInterface::ExternalModel<
          SchedulingBarrierModel<OpT>, OpT> {};
} // namespace

void registerSchedulingBarrierExternalModel(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *) {
    ROCDL::SchedBarrier::attachInterface<
        SchedulingBarrierModel<ROCDL::SchedBarrier>>(*ctx);
    ROCDL::SchedGroupBarrier::attachInterface<
        SchedulingBarrierModel<ROCDL::SchedGroupBarrier>>(*ctx);
  });
}
} // namespace mlir::triton::AMD
