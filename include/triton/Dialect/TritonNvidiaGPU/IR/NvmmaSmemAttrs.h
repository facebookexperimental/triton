#pragma once

#include <optional>
#include <utility>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::nvidia_gpu {

struct NvmmaSmemAttrs {
  unsigned swizzlingByteWidth = 0;
  bool transposed = false;
  bool fp4Padded = false;
};

// Internal helper for creating SMEM descriptors for MMA instructions. The
// nvmmaSmemLL input must map two logical dims to offset/block outputs.
std::optional<std::pair<NvmmaSmemAttrs, LinearLayout>>
getNvmmaSmemAttrs(const LinearLayout &nvmmaSmemLL, unsigned bitwidth);

std::optional<NvmmaSmemAttrs> getNvmmaSmemAttrs(gpu::MemDescType memTy);

// The attrs an MMA smem descriptor reads memTy with, matched from its layout
// as the lowering does, even for an NVMMAShared encoding; nullopt if none.
std::optional<NvmmaSmemAttrs> getMmaOperandSmemAttrs(gpu::MemDescType memTy);

// A mixed-radix register layout cut at its odd register digits into pow2 boxes
// of the pow2 alloc shape. The ldmatrix lowering and the mma.sync legality both
// ask it, so an accepted dot's operands never take the per-element run path.
struct Pow2Chunks {
  LinearLayout layout;
  SmallVector<SmallVector<int32_t>> starts;
};
std::optional<Pow2Chunks> getPow2Chunks(const LinearLayout &regLayout);

} // namespace mlir::triton::nvidia_gpu
