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

// The radix-2 layout an MMA smem descriptor reads ll through: ll itself, or
// for a mixed-radix ll (core matrices at whole-core-matrix strides) its
// offsets at unit coordinates, cut to the tiles of instrShape if given.
std::optional<LinearLayout>
getNvmmaDescriptorLayout(const LinearLayout &ll, unsigned bitwidth,
                         ArrayRef<unsigned> instrShape = {});

// The attrs an MMA smem descriptor reads memTy with, matched from its layout
// as the lowering does, even for an NVMMAShared encoding; nullopt if none.
std::optional<NvmmaSmemAttrs> getMmaOperandSmemAttrs(gpu::MemDescType memTy);

// Whether wgmma can read a [K, n] B with these attrs in tiles of instrN
// columns. An N-major descriptor's swizzle pattern starts at its address, so
// a tile may start inside a pattern only if it also ends there.
bool isMmaV3BTilingLegal(const NvmmaSmemAttrs &attrs, unsigned bitwidth,
                         unsigned instrN, int64_t n);

} // namespace mlir::triton::nvidia_gpu
