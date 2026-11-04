#ifndef TRITON_UNITTEST_TOOLS_PRODUCTLAYOUTS_H
#define TRITON_UNITTEST_TOOLS_PRODUCTLAYOUTS_H

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {

// in -> out identity of a non-power-of-two size: binary digits, then one odd
// digit.
inline LinearLayout exactIdentity1D(int32_t size, StringAttr in,
                                    StringAttr out) {
  int32_t twoPart = LinearLayout::twoPartOf(size);
  LinearLayout::BasesT bases;
  LinearLayout::RadicesT radices;
  for (int32_t bit = 1; bit < twoPart; bit *= 2) {
    bases[in].push_back({bit});
    radices[in].push_back(2);
  }
  bases[in].push_back({twoPart});
  radices[in].push_back(size / twoPart);
  return LinearLayout(std::move(bases), std::move(radices), {{out, size}},
                      /*requireSurjective=*/true);
}

// register and lane inputs onto {dim0, dim1 = 8}; dim0 is Product unless pow2.
inline LinearLayout
makeRegLaneLayout(MLIRContext &ctx,
                  std::vector<std::vector<int32_t>> registerBases,
                  std::vector<int32_t> registerRadices,
                  std::vector<std::vector<int32_t>> laneBases,
                  std::vector<int32_t> laneRadices, int32_t dim0Size) {
  auto S = [&](StringRef str) { return StringAttr::get(&ctx, str); };
  if (laneRadices.empty())
    laneRadices.assign(laneBases.size(), 2);
  LinearLayout::BasesT bases;
  bases[S("register")] = std::move(registerBases);
  bases[S("lane")] = std::move(laneBases);
  LinearLayout::RadicesT radices;
  radices[S("register")] = std::move(registerRadices);
  radices[S("lane")] = std::move(laneRadices);
  return LinearLayout(std::move(bases), std::move(radices),
                      {{S("dim0"), dim0Size}, {S("dim1"), 8}},
                      /*requireSurjective=*/true);
}

} // namespace mlir::triton

#endif // TRITON_UNITTEST_TOOLS_PRODUCTLAYOUTS_H
