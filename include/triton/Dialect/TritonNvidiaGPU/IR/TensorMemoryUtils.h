#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <functional>
#include <optional>

namespace mlir::triton::nvidia_gpu {

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
int getContextualMaxNReg(Operation *op);
struct TMemLdStEncodingInfo {
  TMemAccessAtom atom;
  LinearLayout reps;
  ColumnAction perm;
  int numRegsPerMessage;
  std::optional<uint32_t> secondHalfOffset;
  std::optional<ColumnAction> broadcast = std::nullopt;
  bool unpacked = false;
  unsigned vec = 1;
  bool padding = false;
};

// Return the physical (register, lane, warp) -> (row, col) TMEM layout used by
// tcgen05 load/store lowering. This includes the implicit warp-to-row mapping
// required by the hardware access atoms.
FailureOr<LinearLayout> computeTMemLdStPhysicalLayout(
    RankedTensorType regTy, gpu::MemDescType memTy,
    std::function<InFlightDiagnostic()> emitError = {});

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, gpu::MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {});

// Check whether a tmem_load with register layout `regTy` reading from `memTy`
// can be supported with a fused reduction along the N dimension. This is the
// case when the layout is packed, the per-thread register bases span the full N
// axis and do not advance M. In other words, each thread owns one M coordinate
// and all N values for that coordinate; N is not split across lanes/warps/CTAs,
// and a thread does not hold multiple M rows for this fused reduction.
// `maxnreg` is the contextual register limit and when `emitError` is provided,
// a diagnostic describing the first failing condition is emitted.
bool supportsTMemLoadReduce(RankedTensorType regTy, gpu::MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {});

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_
