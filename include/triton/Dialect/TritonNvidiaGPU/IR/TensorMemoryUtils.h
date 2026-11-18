#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

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
  // An NPOT K of scales moves as its pow2 alloc shape: entry r is the register
  // that holds padded register r, or -1 past K.
  SmallVector<int32_t> npotPaddedRegs;
};

// Why scales of an NPOT MN can't move as pow2 row chunks, or nullopt.
std::optional<std::string>
getNpotTMemScalesUnsupportedReason(int64_t mn, bool pow2K, bool oneCTA);

// The pow2 column chunk that an NPOT TMEM view of N = c * q (q odd) moves q
// times (row chunk for NPOT MN scales), or failure if that plan doesn't cover
// `memTy`.
FailureOr<gpu::MemDescType>
getNpotTMemChunkType(gpu::MemDescType memTy,
                     std::function<InFlightDiagnostic()> emitError = {});

// Map registers, lanes, and all warps to physical TMEM rows and element
// columns. Preserve warp-addressable replicas when the memory layout broadcasts
// rows.
FailureOr<LinearLayout>
computeTMemLdStLayout(RankedTensorType regTy, gpu::MemDescType memTy,
                      std::function<InFlightDiagnostic()> emitError = {});

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, gpu::MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {});

// Return the complete per-message footprint, including vectorization beyond
// the base instruction atom.
FailureOr<LinearLayout> getTMemLdStMessageLayout(MLIRContext *ctx,
                                                 TMemAccessAtom atom,
                                                 bool unpacked,
                                                 int numRegsPerMessage);

// Return the physical tensor-memory word addresses touched by each thread when
// lowering a tmem_load or tmem_store with these types, indexed by
// warp * 32 + lane.
FailureOr<SmallVector<DenseSet<uint32_t>>>
getTMemLdStThreadAddresses(RankedTensorType regTy, gpu::MemDescType memTy,
                           int maxnreg, uint32_t baseAddress = 0);

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
