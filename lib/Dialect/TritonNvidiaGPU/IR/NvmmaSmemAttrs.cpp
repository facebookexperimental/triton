#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

#include <cassert>

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

std::optional<std::pair<NvmmaSmemAttrs, LinearLayout>>
getNvmmaSmemAttrs(const LinearLayout &nvmmaSmemLL, unsigned bitwidth) {
  assert(nvmmaSmemLL.getNumInDims() == 2);
  assert(nvmmaSmemLL.getNumOutDims() == 2);

  auto dims = llvm::to_vector<2>(nvmmaSmemLL.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto offset = StringAttr::get(ctx, "offset");
  auto block = StringAttr::get(ctx, "block");
  assert(nvmmaSmemLL.hasOutDim(offset) && nvmmaSmemLL.hasOutDim(block));
  if (!nvmmaSmemLL.isRadix2()) {
    auto descLL = getNvmmaDescriptorLayout(nvmmaSmemLL, bitwidth);
    return descLL ? getNvmmaSmemAttrs(*descLL, bitwidth) : std::nullopt;
  }

  // TODO: sw=0 is only matched for the MMA "core-matrices" operand form. The
  // other sw=0 interpretation (nvmma_shared<sw=0> as a flat TMA destination) is
  // not supported here.
  for (bool fp4Padded : {false, true}) {
    for (unsigned swizzling : {0u, 32u, 64u, 128u}) {
      for (bool transposed : {false, true}) {
        auto enc = ttg::NVMMASharedEncodingAttr::get(
            ctx, swizzling, transposed, std::max(8u, bitwidth), fp4Padded,
            ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/2));
        auto coreMatrixLL =
            ttg::getCoreMatrixLinearLayout(enc, /*disableSwizzle=*/false);
        auto outDims = llvm::to_vector(coreMatrixLL.getOutDims());
        outDims[0].first = dims[0];
        outDims[1].first = dims[1];
        coreMatrixLL = LinearLayout(coreMatrixLL.getBases(), outDims,
                                    /*requireSurjective=*/false);
        if (bitwidth == 4)
          coreMatrixLL =
              LinearLayout::identity1D(2, offset, dims[1]) * coreMatrixLL;
        if (transposed)
          coreMatrixLL = transposeLinearLayout(coreMatrixLL, {1, 0});
        auto candidateLL = coreMatrixLL.pseudoinvert();
        // Add a trivial block dimension as getReps expects both layouts to
        // have the same outdims
        auto matchLL =
            candidateLL * LinearLayout::identity1D(1, dims[0], block);
        if (getReps(nvmmaSmemLL, matchLL).has_value())
          return std::make_pair(
              NvmmaSmemAttrs{swizzling, transposed, fp4Padded},
              std::move(candidateLL));
      }
    }
  }
  return std::nullopt;
}

std::optional<LinearLayout>
getNvmmaDescriptorLayout(const LinearLayout &ll, unsigned bitwidth,
                         ArrayRef<unsigned> instrShape) {
  if (ll.isRadix2())
    return ll;
  auto dims = llvm::to_vector<2>(ll.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto offset = StringAttr::get(ctx, "offset");
  auto block = StringAttr::get(ctx, "block");
  int32_t offsetIdx = ll.getOutDimIndex(offset);
  int32_t blockIdx = ll.getOutDimIndex(block);
  auto at = [&](int32_t x0, int32_t x1) {
    return ll.apply({{dims[0], x0}, {dims[1], x1}});
  };
  LinearLayout::BasesT bases;
  int32_t maxOffset = 0;
  for (auto [d, dim] : llvm::enumerate(dims)) {
    auto &dimBases = bases[dim];
    for (int32_t x = 1; x < ll.getInDimSize(dim); x *= 2) {
      auto outs = d == 0 ? at(x, 0) : at(0, x);
      if (outs[blockIdx].second != 0)
        return std::nullopt;
      maxOffset = std::max(maxOffset, outs[offsetIdx].second);
      dimBases.push_back({outs[0].second, outs[1].second});
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> outDims = ll.getOutDims();
  outDims[offsetIdx].second = llvm::NextPowerOf2(maxOffset);
  outDims[blockIdx].second = 1;
  auto descLL = LinearLayout::tryCreate(std::move(bases), outDims,
                                        /*requireSurjective=*/false);
  if (!descLL)
    return std::nullopt;
  auto attrsAndTile = getNvmmaSmemAttrs(*descLL, bitwidth);
  if (!attrsAndTile)
    return std::nullopt;
  // The descriptor adds whole core matrices at its LBO and SBO, so ll must be
  // the core matrix plus those strides times the tile index, everywhere.
  const LinearLayout &tile = attrsAndTile->second;
  int32_t tile0 = tile.getInDimSize(dims[0]);
  int32_t tile1 = tile.getInDimSize(dims[1]);
  auto stride = [&](int32_t d, int32_t size) {
    if (size >= ll.getInDimSize(dims[d]))
      return 0;
    return descLL->getBasis(dims[d], llvm::Log2_32(size), offset);
  };
  int32_t stride0 = stride(0, tile0), stride1 = stride(1, tile1);
  // Whole core matrices keep the swizzle phase the descriptor assumes.
  int64_t tileBits = int64_t(tile0) * tile1 * bitwidth;
  if (int64_t(stride0) * bitwidth % tileBits != 0 ||
      int64_t(stride1) * bitwidth % tileBits != 0)
    return std::nullopt;
  for (int32_t x0 = 0; x0 < ll.getInDimSize(dims[0]); ++x0) {
    for (int32_t x1 = 0; x1 < ll.getInDimSize(dims[1]); ++x1) {
      int32_t core =
          descLL
              ->apply({{dims[0], x0 % tile0}, {dims[1], x1 % tile1}})[offsetIdx]
              .second;
      auto outs = at(x0, x1);
      if (outs[blockIdx].second != 0 ||
          outs[offsetIdx].second !=
              core + x0 / tile0 * stride0 + x1 / tile1 * stride1)
        return std::nullopt;
    }
  }
  // Beyond one instruction's tiles the strides add, which a radix-2 layout
  // cannot express; the per-tile offsets come from ll itself.
  for (auto [d, instr] : llvm::enumerate(instrShape))
    descLL = descLL->resizeInDim(dims[d],
                                 std::max<int32_t>(tile.getInDimSize(dims[d]),
                                                   llvm::PowerOf2Ceil(instr)));
  return descLL;
}

std::optional<NvmmaSmemAttrs> getNvmmaSmemAttrs(ttg::MemDescType memTy) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(memTy.getEncoding()))
    return NvmmaSmemAttrs{nvmma.getSwizzlingByteWidth(), nvmma.getTransposed(),
                          nvmma.getFp4Padded()};
  return getMmaOperandSmemAttrs(memTy);
}

std::optional<NvmmaSmemAttrs> getMmaOperandSmemAttrs(ttg::MemDescType memTy) {
  auto ll = ttg::toLinearLayout(memTy).pseudoinvert();
  unsigned bitwidth = memTy.getElementType().getIntOrFloatBitWidth();
  auto attrsAndCandidate = getNvmmaSmemAttrs(ll, bitwidth);
  if (!attrsAndCandidate)
    return std::nullopt;
  return attrsAndCandidate->first;
}

bool isMmaV3BTilingLegal(const NvmmaSmemAttrs &attrs, unsigned bitwidth,
                         unsigned instrN, int64_t n) {
  if (attrs.transposed)
    return true;
  int64_t pattern = std::max(attrs.swizzlingByteWidth, 16u) * 8 / bitwidth;
  for (int64_t start = instrN; start < n; start += instrN)
    if (start % pattern != 0 && start % pattern + instrN > pattern)
      return false;
  return true;
}

std::optional<Pow2Chunks> getPow2Chunks(const LinearLayout &regLayout) {
  auto kReg = StringAttr::get(regLayout.getInDimNames().begin()->getContext(),
                              "register");
  int32_t numBinary = regLayout.getNumLeadingRadix2Digits(kReg);
  auto bases = regLayout.getBases();
  bases[kReg].resize(numBinary);
  SmallVector<std::pair<StringAttr, int32_t>> box;
  for (auto [d, dim] : llvm::enumerate(regLayout.getOutDimNames())) {
    int32_t bits = 0;
    for (const auto &inBases : llvm::make_second_range(bases))
      for (const auto &basis : inBases)
        bits |= basis[d];
    box.push_back({dim, static_cast<int32_t>(llvm::NextPowerOf2(bits))});
    if (box.back().second >
        LinearLayout::twoPartOf(regLayout.getOutDimSize(dim)))
      return std::nullopt;
  }
  for (StringAttr inDim : regLayout.getInDimNames())
    if (inDim != kReg &&
        !llvm::all_of(regLayout.getInputRadices(inDim),
                      [](int32_t radix) { return radix == 2; }))
      return std::nullopt;
  auto layout = LinearLayout::tryCreate(std::move(bases), box,
                                        /*requireSurjective=*/true);
  if (!layout)
    return std::nullopt;
  Pow2Chunks chunks{std::move(*layout), {}};
  for (int32_t reg = 0; reg < regLayout.getInDimSize(kReg);
       reg += 1 << numBinary) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    for (StringAttr inDim : regLayout.getInDimNames())
      ins.push_back({inDim, inDim == kReg ? reg : 0});
    // A start off its box's alignment would overlap the box bits it XORs.
    auto &start = chunks.starts.emplace_back();
    for (auto [d, out] : llvm::enumerate(regLayout.apply(ins))) {
      if (out.second % box[d].second != 0)
        return std::nullopt;
      start.push_back(out.second);
    }
  }
  return chunks;
}

} // namespace mlir::triton::nvidia_gpu
