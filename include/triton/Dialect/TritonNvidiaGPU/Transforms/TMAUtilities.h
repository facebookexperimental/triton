#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TMAUtils.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp6Padded(Attribute encoding) {
  auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding);
  return mmaEnc && mmaEnc.isPaddedFp6();
}

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

int64_t getDescriptorLoadBytes(Operation *op, RankedTensorType tensorType,
                               Value desc);
int64_t getDescriptorLoadBytes(triton::gpu::MemDescType memDescType);

bool hasCGABroadcast(gpu::MemDescType memDescType);

Value sextI16ToI32Indices(Value indices, OpBuilder &builder, Location loc);

FailureOr<int> getTMASwizzleMode(Location loc, triton::TensorDescInterface ty);
FailureOr<int> getTMAElementType(Location loc, triton::TensorDescInterface ty);

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

} // namespace mlir::triton::nvidia_gpu
