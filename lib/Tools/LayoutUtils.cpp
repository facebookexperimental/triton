#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <array>

namespace mlir::triton {

namespace {

// Bit offsets of each out dim's binary part in one uint64, sized for the wider
// of the two layouts (same out-dim order); the last entry is the total width.
std::optional<SmallVector<int32_t>> getBinaryOffsets(const LinearLayout &a,
                                                     const LinearLayout &b) {
  SmallVector<int32_t> offsets = {0};
  for (StringAttr outDim : a.getOutDimNames()) {
    auto width = [&](const LinearLayout &layout) {
      return static_cast<int32_t>(llvm::countr_zero(
          static_cast<uint32_t>(layout.getOutDimSize(outDim))));
    };
    offsets.push_back(offsets.back() + std::max(width(a), width(b)));
  }
  if (offsets.back() > 64)
    return std::nullopt;
  return offsets;
}

// The radix-2 digits of a Binary or mixed-radix layout as GF(2) columns in
// in-dim and digit order. A column in the span of earlier columns is free, as
// in getFreeVariableMasks, and its kernel vector touches only earlier pivots.
struct BinaryColumns {
  SmallVector<std::pair<StringAttr, int32_t>> digits;
  SmallVector<uint64_t> images;
  SmallVector<int32_t> freeColumns;
  SmallVector<uint64_t> kernels;
  std::array<uint64_t, 64> pivotImage{};
  std::array<uint64_t, 64> pivotCombo{};

  uint64_t reduce(uint64_t &image) const {
    uint64_t combo = 0;
    for (int bit = 63; bit >= 0; --bit) {
      if ((image >> bit & 1) && pivotImage[bit]) {
        image ^= pivotImage[bit];
        combo ^= pivotCombo[bit];
      }
    }
    return combo;
  }

  // The preimage with every free column zero, or nullopt outside the image.
  std::optional<uint64_t> preimage(uint64_t image) const {
    uint64_t combo = reduce(image);
    if (image != 0)
      return std::nullopt;
    return combo;
  }
};

std::optional<BinaryColumns> getBinaryColumns(const LinearLayout &layout,
                                              ArrayRef<int32_t> offsets) {
  BinaryColumns columns;
  for (StringAttr inDim : layout.getInDimNames()) {
    for (auto [pos, radix] : llvm::enumerate(layout.getInputRadices(inDim))) {
      if (radix != 2)
        continue;
      if (columns.digits.size() == 64)
        return std::nullopt;
      uint64_t image = 0;
      for (auto [index, outDim] : llvm::enumerate(layout.getOutDimNames())) {
        int32_t binary = layout.getBasisContribution(inDim, pos, outDim).binary;
        if (binary != 0)
          image |= static_cast<uint64_t>(binary) << offsets[index];
      }
      uint64_t self = uint64_t{1} << columns.digits.size();
      columns.digits.push_back({inDim, static_cast<int32_t>(pos)});
      columns.images.push_back(image);
      uint64_t combo = columns.reduce(image) ^ self;
      if (image == 0) {
        columns.freeColumns.push_back(columns.digits.size() - 1);
        columns.kernels.push_back(combo);
        continue;
      }
      int lead = 63 - llvm::countl_zero(image);
      columns.pivotImage[lead] = image;
      columns.pivotCombo[lead] = combo;
    }
  }
  return columns;
}

// The odd digit owning each out dim's odd coordinate, with its coefficient.
struct OddOwner {
  StringAttr inDim;
  int32_t pos;
  int32_t coefficient;
};

// Returns nullopt if some odd digit owns no coordinate (a zero digit).
std::optional<SmallVector<std::optional<OddOwner>>>
getOddOwners(const LinearLayout &layout) {
  SmallVector<std::optional<OddOwner>> owners(layout.getNumOutDims());
  for (StringAttr inDim : layout.getInDimNames()) {
    for (auto [pos, radix] : llvm::enumerate(layout.getInputRadices(inDim))) {
      if (radix == 2)
        continue;
      bool owns = false;
      for (auto [index, outDim] : llvm::enumerate(layout.getOutDimNames())) {
        int32_t odd = layout.getBasisContribution(inDim, pos, outDim).odd;
        if (odd == 0)
          continue;
        owners[index] = OddOwner{inDim, static_cast<int32_t>(pos), odd};
        owns = true;
      }
      if (!owns)
        return std::nullopt;
    }
  }
  return owners;
}

// The inverse of `a` modulo `m` by extended Euclid, or nullopt for a non-unit.
std::optional<int64_t> inverseUnit(int64_t a, int64_t m) {
  // Invariant: s * a == r (mod m) for both (r, s) pairs.
  int64_t r0 = m, s0 = 0, r1 = a % m, s1 = 1;
  while (r1 != 0) {
    int64_t q = r0 / r1;
    std::tie(r0, r1) = std::make_pair(r1, r0 - q * r1);
    std::tie(s0, s1) = std::make_pair(s1, s0 - q * s1);
  }
  if (r0 != 1)
    return std::nullopt;
  return (s0 % m + m) % m;
}

bool isRegisterOrLane(StringAttr inDim) {
  return inDim == "register" || inDim == "lane";
}

bool hasOddOutsideRegister(const LinearLayout &layout) {
  return llvm::any_of(layout.getInDimNames(), [&](StringAttr dim) {
    return dim != "register" &&
           llvm::any_of(layout.getInputRadices(dim),
                        [](int32_t radix) { return radix != 2; });
  });
}

} // namespace

bool coversAllLogicalOutputs(const LinearLayout &layout,
                             ArrayRef<int64_t> logicalShape) {
  if (logicalShape.size() != layout.getNumOutDims())
    return false;
  if (logicalShape.empty())
    return true;
  auto outDimNames = llvm::to_vector(layout.getOutDimNames());
  if (outDimNames != standardOutDimNames(outDimNames.front().getContext(),
                                         static_cast<int>(logicalShape.size())))
    return false;
  auto offsets = getBinaryOffsets(layout, layout);
  if (!offsets)
    return false;
  auto columns = getBinaryColumns(layout, *offsets);
  if (!columns)
    return false;
  // The image is the binary span plus each owned odd coordinate's whole odd
  // part, so the box needs a preimage of each binary unit below its extent,
  // and an odd owner once the extent passes the binary part.
  for (auto [d, outDim] : llvm::enumerate(outDimNames)) {
    int64_t extent = logicalShape[d];
    int32_t size = layout.getOutDimSize(outDim);
    if (extent <= 0 || extent > size)
      return false;
    int32_t binarySize = LinearLayout::twoPartOf(size);
    if (extent > binarySize) {
      bool oddOwned = false;
      for (StringAttr inDim : layout.getInDimNames())
        for (int32_t i = 0; i < layout.getInDimSizeLog2(inDim); ++i)
          oddOwned |= layout.getInputRadix(inDim, i) != 2 &&
                      layout.getBasisContribution(inDim, i, outDim).odd != 0;
      if (!oddOwned)
        return false;
      extent = binarySize;
    }
    for (int32_t bit = 0; (int64_t{1} << bit) < extent; ++bit)
      if (!columns->preimage(uint64_t{1} << ((*offsets)[d] + bit)))
        return false;
  }
  return true;
}

std::optional<LinearLayout>
alignMixedRadixConversionOutDims(const LinearLayout &srcLayout,
                                 const LinearLayout &dstLayout,
                                 ArrayRef<int64_t> logicalShape) {
  auto outDims = to_vector(srcLayout.getOutDimNames());
  if (outDims.size() != logicalShape.size() ||
      dstLayout.getNumOutDims() != srcLayout.getNumOutDims())
    return std::nullopt;
  if (!outDims.empty() &&
      outDims != standardOutDimNames(outDims.front().getContext(),
                                     static_cast<int>(outDims.size())))
    return std::nullopt;
  if (!llvm::all_of(outDims,
                    [&](StringAttr dim) { return dstLayout.hasOutDim(dim); }))
    return std::nullopt;
  return dstLayout.transposeOuts(outDims);
}

bool hasMixedRadixConversionInputDims(const LinearLayout &layout) {
  return llvm::all_of(layout.getInDimNames(), [&](StringAttr dim) {
    return isRegisterOrLane(dim) || dim == "warp" ||
           (dim == "block" && layout.getInDimSize(dim) == 1);
  });
}

MixedRadixConversionScope
classifyMixedRadixConversionScope(const LinearLayout &srcLayout,
                                  const LinearLayout &dstLayout,
                                  ArrayRef<int64_t> logicalShape) {
  auto offsets = getBinaryOffsets(srcLayout, dstLayout);
  if (!offsets || hasOddOutsideRegister(srcLayout) ||
      hasOddOutsideRegister(dstLayout))
    return MixedRadixConversionScope::Block;
  // Coverage bounds the columns, so both exist.
  auto src = *getBinaryColumns(srcLayout, *offsets);
  auto dst = *getBinaryColumns(dstLayout, *offsets);

  auto outDims = to_vector(srcLayout.getOutDimNames());
  // Odd digits sit in registers, so the lane and warp owners are GF(2)-linear
  // in the low output bits and the destination kernel. Checking one generator
  // per in-bounds bit and kernel vector is exact.
  using Owner = std::pair<uint64_t, uint64_t>;
  auto owner = [](const BinaryColumns &columns, uint64_t combo) {
    Owner laneAndWarp{0, 0};
    for (auto [column, digit] : llvm::enumerate(columns.digits)) {
      if (!(combo >> column & 1))
        continue;
      if (digit.first == "lane")
        laneAndWarp.first ^= uint64_t{1} << digit.second;
      else if (digit.first == "warp")
        laneAndWarp.second ^= uint64_t{1} << digit.second;
    }
    return laneAndWarp;
  };
  MixedRadixConversionScope scope = MixedRadixConversionScope::Register;
  auto widen = [&](Owner source, Owner destination) {
    if (source.second != destination.second)
      scope = MixedRadixConversionScope::Block;
    else if (source.first != destination.first &&
             scope == MixedRadixConversionScope::Register)
      scope = MixedRadixConversionScope::Warp;
  };
  for (uint64_t kernel : dst.kernels)
    widen({0, 0}, owner(dst, kernel));
  for (auto [index, extent] : llvm::enumerate(logicalShape)) {
    int32_t width = (*offsets)[index + 1] - (*offsets)[index];
    for (int32_t bit = 0; bit < width && (int64_t{1} << bit) < extent; ++bit) {
      auto image = [&, dim = index](const LinearLayout &layout) -> uint64_t {
        int32_t size = layout.getOutDimSize(outDims[dim]);
        if (bit >= llvm::countr_zero(static_cast<uint32_t>(size)))
          return 0;
        return uint64_t{1} << ((*offsets)[dim] + bit);
      };
      // Coverage gives each in-bounds bit a preimage.
      widen(owner(src, *src.preimage(image(srcLayout))),
            owner(dst, *dst.preimage(image(dstLayout))));
    }
  }
  return scope;
}

std::optional<MixedRadixConversionInputMap>
mapMixedRadixConversionInputs(const LinearLayout &a, const LinearLayout &b) {
  auto inDims = to_vector(b.getInDimNames());
  auto registerDim = llvm::find(inDims, "register");
  auto laneDim = llvm::find(inDims, "lane");
  if (a.getOutDims() != b.getOutDims() ||
      !llvm::equal(a.getInDimNames(), inDims) || registerDim == inDims.end() ||
      laneDim == inDims.end() || hasOddOutsideRegister(a) ||
      hasOddOutsideRegister(b) || llvm::any_of(inDims, [&](StringAttr dim) {
        return a.getInDimSize(dim) != b.getInDimSize(dim);
      }))
    return std::nullopt;
  auto offsets = getBinaryOffsets(a, b);
  if (!offsets)
    return std::nullopt;
  auto aColumns = getBinaryColumns(a, *offsets);
  auto bColumns = getBinaryColumns(b, *offsets);
  auto aOwners = getOddOwners(a);
  auto bOwners = getOddOwners(b);
  // Equal column and kernel counts give equal ranks, so a preimage of every
  // `b` column below proves the binary images equal.
  if (!aColumns || !bColumns || !aOwners || !bOwners ||
      aColumns->digits.size() != bColumns->digits.size() ||
      aColumns->kernels.size() != bColumns->kernels.size())
    return std::nullopt;

  // values[inDim][pos]: the L output that `b`'s digit maps to. Each in dim's
  // radix-2 `a` digits, in order, are the low bits of that output.
  llvm::SmallDenseMap<StringAttr, SmallVector<int32_t>> values;
  for (StringAttr inDim : inDims)
    values[inDim].assign(b.getInDimSizeLog2(inDim), 0);
  SmallVector<int32_t> aBit;
  for (auto [column, digit] : llvm::enumerate(aColumns->digits)) {
    bool sameDim =
        column > 0 && aColumns->digits[column - 1].first == digit.first;
    aBit.push_back(sameDim ? aBit.back() + 1 : 0);
  }
  // Binary digits: the canonical preimage; the k-th free `b` column takes the
  // k-th `a` kernel vector, so the map is a bijection.
  SmallVector<uint64_t> columnMap;
  for (uint64_t image : bColumns->images) {
    auto preimage = aColumns->preimage(image);
    if (!preimage)
      return std::nullopt;
    columnMap.push_back(*preimage);
  }
  for (auto [k, column] : llvm::enumerate(bColumns->freeColumns))
    columnMap[column] ^= aColumns->kernels[k];
  for (auto [column, digit] : llvm::enumerate(bColumns->digits)) {
    auto [inDim, pos] = digit;
    for (auto [aColumn, aDigit] : llvm::enumerate(aColumns->digits)) {
      if (!(columnMap[column] >> aColumn & 1))
        continue;
      if (aDigit.first != inDim)
        return std::nullopt;
      values[inDim][pos] |= 1 << aBit[aColumn];
    }
  }
  // Odd digits: both own each odd coordinate from a register, and b's
  // coefficient is f times a's, so the digit maps to f on L's top digit.
  StringAttr kReg = *registerDim;
  int64_t registerTwoPart = LinearLayout::twoPartOf(a.getInDimSize(kReg));
  for (auto [index, outDim] : llvm::enumerate(a.getOutDimNames())) {
    const auto &aOwner = (*aOwners)[index];
    const auto &bOwner = (*bOwners)[index];
    if (aOwner.has_value() != bOwner.has_value())
      return std::nullopt;
    if (!aOwner)
      continue;
    int32_t size = a.getOutDimSize(outDim);
    int64_t oddPart = size / LinearLayout::twoPartOf(size);
    auto inverse = inverseUnit(aOwner->coefficient, oddPart);
    if (!inverse)
      return std::nullopt;
    int64_t factor = *inverse * bOwner->coefficient % oddPart;
    values[kReg][bOwner->pos] = static_cast<int32_t>(factor * registerTwoPart);
  }
  // Every other input dimension must map to itself.
  for (StringAttr inDim : inDims) {
    if (isRegisterOrLane(inDim))
      continue;
    for (auto [pos, value] : llvm::enumerate(values[inDim]))
      if (value != 1 << pos)
        return std::nullopt;
  }

  auto makeFactor = [&](StringAttr dim) -> std::optional<LinearLayout> {
    ArrayRef<int32_t> radices = b.getInputRadices(dim);
    LinearLayout::BasesT bases;
    LinearLayout::RadicesT inputRadices;
    auto &dimBases = bases[dim];
    for (int32_t value : values[dim])
      dimBases.push_back({value});
    inputRadices[dim].assign(radices.begin(), radices.end());
    // Surjective onto an equal-size input: a bijection. Two odd register
    // digits share one odd part, which the mixed-radix checks reject.
    return LinearLayout::tryCreate(std::move(bases), std::move(inputRadices),
                                   {{dim, a.getInDimSize(dim)}},
                                   /*requireSurjective=*/true);
  };
  auto registerMap = makeFactor(kReg);
  auto laneMap = makeFactor(*laneDim);
  if (!registerMap || !laneMap)
    return std::nullopt;

  // registerOrder: output digit j, a's flat digit j, is input digit action[j].
  ArrayRef<int32_t> radices = a.getInputRadices(kReg);
  SmallVector<int32_t> oddLastRadices;
  llvm::copy_if(radices, std::back_inserter(oddLastRadices),
                [](int32_t radix) { return radix == 2; });
  SmallVector<size_t> action;
  size_t binary = 0, odd = oddLastRadices.size();
  llvm::copy_if(radices, std::back_inserter(oddLastRadices),
                [](int32_t radix) { return radix != 2; });
  for (int32_t radix : radices)
    action.push_back(radix == 2 ? binary++ : odd++);
  return MixedRadixConversionInputMap{
      std::move(*registerMap), std::move(*laneMap),
      ColumnAction(action, kReg, radices.size(), oddLastRadices)};
}

static bool checkSquareSublayout(const LinearLayout &ll,
                                 ArrayRef<StringAttr> dimNames,
                                 function_ref<bool(int, int32_t)> checkBasis) {
  // The empty layout is the identity
  if (dimNames.size() == 0) {
    return true;
  }
  // Check that the input-output sizes are the same
  LinearLayout sl = ll.sublayout(dimNames, dimNames);
  for (StringAttr dim : dimNames) {
    if (ll.getInDimSize(dim) != ll.getOutDimSize(dim)) {
      return false;
    }
  }
  // Once the inputs and output dimensions are the same, we can just check
  // that the basis for the single remaining dimension is the identity.
  sl = sl.flattenIns().flattenOuts();
  const auto &inDimBases = sl.getBases().begin()->second;
  for (auto [b, basis] : llvm::enumerate(inDimBases)) {
    if (!checkBasis(b, basis[0])) {
      return false;
    }
  }
  return true;
}

bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames) {
  return checkSquareSublayout(
      ll, dimNames, [](int b, int32_t basis) { return basis == (1 << b); });
}

LinearLayout invertAndComposeLocal(const LinearLayout &A, const LinearLayout &B,
                                   ArrayRef<StringAttr> localDims) {
  for (StringAttr dim : localDims) {
    assert(A.hasInDim(dim) && B.hasInDim(dim));
    assert(A.getInDimSize(dim) <= B.getInDimSize(dim));
  }

  // Zero-extend A's bases where it is smaller than B along a local dim, so
  // the composition below covers B's full local range.
  auto localA = A;
  {
    auto bases = A.getBases();
    bool extended = false;
    for (StringAttr dim : localDims) {
      if (A.getInDimSize(dim) < B.getInDimSize(dim)) {
        bases[dim].resize(B.getBases().at(dim).size(),
                          std::vector<int32_t>(A.getNumOutDims(), 0));
        extended = true;
      }
    }
    if (extended)
      localA = LinearLayout(std::move(bases), A.getOutDims(),
                            /*requireSurjective=*/false);
  }
  auto cvt = B.invertAndCompose(localA);
  auto bases = cvt.getBases();
  for (StringAttr dim : localDims) {
    // A zero source basis leaves its output coordinate free. Choose the
    // corresponding input bit to keep the replicated value local.
    int outIdx = cvt.getOutDimIndex(dim);
    uint64_t nonZeroBases = getInputBasisMask(
        localA, dim, llvm::to_vector(localA.getOutDimNames()));
    for (auto &inDimBases : llvm::make_second_range(bases))
      for (auto &basis : inDimBases)
        basis[outIdx] &= nonZeroBases;
    for (auto [i, basis] : llvm::enumerate(bases[dim]))
      basis[outIdx] |= (uint64_t{1} << i) & ~nonZeroBases;
  }
  return LinearLayout(std::move(bases), cvt.getOutDims(),
                      /*requireSurjective=*/false);
}

// Under TRITON_ALLOW_NPOT an NPOT extent is held by its pow2 alloc shape; the
// coordinates past the extent are dead.
static int64_t getPow2AllocSize(int64_t size) {
  if (!mlir::triton::tools::isNpotAllowed())
    return size;
  return llvm::PowerOf2Ceil(size);
}

LinearLayout
ensureLayoutNotLargerThan(const LinearLayout &layout,
                          const llvm::SmallDenseMap<StringAttr, int64_t> &shape,
                          bool broadcastRegisters) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }
  MLIRContext *ctx = shape.begin()->first.getContext();
  auto bases = layout.getBases();
  auto kRegister = StringAttr::get(ctx, "register");
  std::set<int32_t> broadcastedDims;

  for (auto outDim : llvm::enumerate(layout.getOutDimNames())) {
    auto outDimName = outDim.value();
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = getPow2AllocSize(shape.lookup(outDimName));
    if (actualSize <= desiredSize) {
      continue;
    }
    assert(actualSize % desiredSize == 0);
    // <inDimName, basisIdx, outValue>
    std::vector<std::tuple<StringAttr, int, int>> sortedBases;
    for (auto [inDimName, basis] : bases) {
      for (size_t basisIdx = 0; basisIdx < basis.size(); basisIdx++) {
        auto outValue = basis[basisIdx][outDim.index()];
        if (outValue == 0) {
          continue;
        }
        assert(llvm::isPowerOf2_32(outValue));
        sortedBases.emplace_back(inDimName, basisIdx, outValue);
      }
    }
    // From the largest basis to the smallest.
    llvm::sort(sortedBases,
               [](auto a, auto b) { return std::get<2>(a) > std::get<2>(b); });
    for (size_t i = 0; i < sortedBases.size() && actualSize > desiredSize;) {
      int outValue = std::get<2>(sortedBases[i]);
      do {
        auto inDimName = std::get<0>(sortedBases[i]);
        auto basisIdx = std::get<1>(sortedBases[i++]);
        if (!broadcastRegisters && inDimName == kRegister) {
          broadcastedDims.insert(basisIdx);
        } else {
          bases[inDimName][basisIdx][outDim.index()] = 0;
        }
      } while (i < sortedBases.size() &&
               std::get<2>(sortedBases[i]) == outValue);
      // Duplicate bases encode the same output bit and halve actualSize only
      // after every copy has been removed.
      actualSize >>= 1;
    }
  }
  auto radices = layout.getInputRadices();
  if (!broadcastRegisters) {
    std::vector<std::vector<int32_t>> newBasesRegister;
    std::vector<int32_t> newRadicesRegister;
    for (auto [idx, basis] : llvm::enumerate(bases[kRegister])) {
      if (broadcastedDims.find(idx) == broadcastedDims.end()) {
        newBasesRegister.push_back(std::move(basis));
        newRadicesRegister.push_back(radices[kRegister][idx]);
      }
    }
    bases[kRegister] = std::move(newBasesRegister);
    radices[kRegister] = std::move(newRadicesRegister);
  }
  auto outDims = layout.getOutDims();
  for (auto &[outDim, outDimSize] : outDims) {
    outDimSize =
        std::min<int32_t>(outDimSize, getPow2AllocSize(shape.lookup(outDim)));
  }

  return LinearLayout(std::move(bases), std::move(radices), std::move(outDims),
                      /*requireSurjective=*/false);
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along its most-minor dimension ("register" for register layouts, "offset" for
// shared layouts).
//
// This function is invariant to the order of the layout's input dimensions, but
// it cares about the order of the output dims, which should be minor-to-major.
LinearLayout ensureLayoutNotSmallerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }

  StringAttr kDim = *layout.getInDimNames().begin();
  assert(kDim == "register" || kDim == "offset");

  LinearLayout ret = layout;
  for (StringAttr outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = getPow2AllocSize(shape.lookup(outDimName));
    assert(actualSize > desiredSize || desiredSize % actualSize == 0);
    ret *= LinearLayout::identity1D(desiredSize / actualSize, kDim, outDimName);
    assert(ret.getOutDimSize(outDimName) >= desiredSize);
  }
  return ret;
}

// Returns ["dim0", "dim1", ..., "dim<rank-1>"].
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank) {
  SmallVector<StringAttr> ret;
  for (int i = 0; i < rank; i++) {
    ret.push_back(StringAttr::get(ctx, "dim" + llvm::Twine(i)));
  }
  return ret;
}

// Returns [("dim0", dstShape[0]), ("dim1", dstShape[1]), ...,
// ("dim<rank-1>", dstShape[rank-1])].
SmallVector<std::pair<StringAttr, int32_t>>
standardOutDimPairs(MLIRContext *ctx, ArrayRef<int64_t> dstShape) {
  auto newRank = dstShape.size();
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto [dim, size] :
       llvm::zip(standardOutDimNames(ctx, newRank), dstShape)) {
    newOutDims.emplace_back(dim, size);
  }
  return newOutDims;
}

// Returns a 1D -> ND layout into [dim0, dim1, ...] that's equivalent to
// creating a 1D -> 1D mapping of size product(shape) and then reshaping to
// permute(shape, order).
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  MLIRContext *ctx = inDimName.getContext();
  auto rank = shape.size();

  // The order in triton is written wrt. [dim0, dim1, ...].
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
}

LinearLayout zerosLike(const LinearLayout &layout) {
  auto bases = layout.getBases();
  for (auto &basis : bases) {
    for (auto &vec : basis.second) {
      for (auto &val : vec) {
        val = 0;
      }
    }
  }

  return LinearLayout(std::move(bases), layout.getOutDims(),
                      /*requireSurjective=*/false);
}

uint32_t getOutputBasisMask(const LinearLayout &layout,
                            ArrayRef<StringAttr> inDims, StringAttr outDim) {
  assert(layout.hasOutDim(outDim));
  unsigned outIdx = layout.getOutDimIndex(outDim);
  uint32_t mask = 0;
  for (StringAttr inDim : inDims) {
    assert(layout.hasInDim(inDim));
    for (const auto &basis : layout.getBases().lookup(inDim))
      mask |= uint32_t(basis[outIdx]);
  }
  return mask;
}

uint64_t getInputBasisMask(const LinearLayout &layout, StringAttr inDim,
                           ArrayRef<StringAttr> outDims) {
  assert(layout.hasInDim(inDim));
  SmallVector<unsigned> outIndices;
  for (StringAttr outDim : outDims) {
    assert(layout.hasOutDim(outDim));
    outIndices.push_back(layout.getOutDimIndex(outDim));
  }

  uint64_t mask = 0;
  for (const auto &indexedBasis :
       llvm::enumerate(layout.getBases().lookup(inDim))) {
    const auto &basis = indexedBasis.value();
    if (llvm::any_of(outIndices,
                     [&](unsigned outIdx) { return basis[outIdx] != 0; }))
      mask |= uint64_t{1} << indexedBasis.index();
  }
  return mask;
}

IdentityFactor factorMaximalIdentityPrefix(const LinearLayout &layout,
                                           StringAttr inDim, StringAttr outDim,
                                           int32_t maxSize) {
  assert(layout.hasInDim(inDim) && layout.hasOutDim(outDim));
  assert(maxSize > 0 && llvm::isPowerOf2_32(maxSize));

  IdentityFactor result{1, layout};
  for (int32_t size = 2; size <= maxSize;) {
    auto quotient =
        divideLeft(layout, LinearLayout::identity1D(size, inDim, outDim));
    if (!quotient)
      break;
    result = {size, std::move(*quotient)};
    if (size == maxSize)
      break;
    size *= 2;
  }
  return result;
}

LinearLayout renameLinearLayoutDims(
    const LinearLayout &layout,
    ArrayRef<std::pair<StringAttr, StringAttr>> inDimRenames,
    ArrayRef<std::pair<StringAttr, StringAttr>> outDimRenames) {
  auto getRenames = [](auto dimNames, auto renames) {
    llvm::SmallDenseSet<StringAttr> existing(dimNames.begin(), dimNames.end());
    llvm::DenseMap<StringAttr, StringAttr> result;
    for (auto [oldDim, newDim] : renames) {
      assert(existing.contains(oldDim) && "dimension to rename does not exist");
      assert(!result.contains(oldDim) && "dimension renamed more than once");
      result.try_emplace(oldDim, newDim);
    }

    llvm::SmallDenseSet<StringAttr> finalNames;
    for (StringAttr dim : dimNames) {
      StringAttr renamed = result.lookup(dim);
      StringAttr finalName = renamed ? renamed : dim;
      assert(!finalNames.contains(finalName) &&
             "renaming creates duplicate dimensions");
      finalNames.insert(finalName);
    }
    return result;
  };

  auto inRenames = getRenames(layout.getInDimNames(), inDimRenames);
  auto outRenames = getRenames(layout.getOutDimNames(), outDimRenames);

  LinearLayout::BasesT bases;
  LinearLayout::RadicesT radices;
  for (auto [inDim, inDimBases] : layout.getBases()) {
    StringAttr renamed = inRenames.lookup(inDim);
    bases.insert({renamed ? renamed : inDim, inDimBases});
    radices.insert(
        {renamed ? renamed : inDim, layout.getInputRadices().lookup(inDim)});
  }
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (auto [outDim, size] : layout.getOutDims()) {
    StringAttr renamed = outRenames.lookup(outDim);
    outDims.push_back({renamed ? renamed : outDim, size});
  }
  return LinearLayout(std::move(bases), std::move(radices), outDims,
                      /*requireSurjective=*/layout.isSurjective());
}

std::optional<ColumnAction> regPermForDivide(const LinearLayout &A,
                                             const LinearLayout &B, bool left) {
  // We can implement this generically for any dimension, but for now we only do
  // it for regs to keep the API simpler
  assert(A.getNumInDims() != 0);
  auto kReg = *A.getInDimNames().begin();
  assert(kReg.str() == "register");
  assert(B.getNumInDims() != 0);
  assert(kReg == *B.getInDimNames().begin());

  // We broadcast B to have the same number of out dims as A.
  LinearLayout broadcast;
  for (StringAttr out : A.getOutDimNames()) {
    broadcast *= LinearLayout::identity1D(1, kReg, out);
  }
  auto BBroadcast = broadcast * B;

  // Retrieve the register bases from A and B.
  const auto &ARegBases = A.getBases().lookup(kReg);
  const auto &BRegBases = BBroadcast.getBases().lookup(kReg);

  llvm::DenseMap<StringAttr, unsigned> log2QuotSize;
  for (StringAttr out : A.getOutDimNames()) {
    log2QuotSize[out] =
        A.getOutDimSizeLog2(out) - BBroadcast.getOutDimSizeLog2(out);
    if (log2QuotSize[out] < 0)
      return std::nullopt;
  }

  auto multiplyByTileSize =
      [&](ArrayRef<int32_t> bBasis) -> std::vector<int32_t> {
    std::vector<int32_t> result;
    assert(bBasis.size() == A.getNumOutDims());
    for (auto [dim, b] : llvm::zip(A.getOutDimNames(), bBasis)) {
      result.push_back(b << log2QuotSize.lookup(dim));
    }
    return result;
  };

  // Compute the permutation order:
  // For each basis in B (in order), find its index in A (using each index at
  // most once). We make sure we use each index at most once in case B
  // broadcasts (weird case, but better safe than sorry).
  SmallVector<size_t> bIndices;
  SmallVector<bool> used(ARegBases.size(), false);
  for (auto bB : BRegBases) {
    bool found = false;
    if (!left)
      bB = multiplyByTileSize(bB);

    for (size_t j = 0; j < ARegBases.size(); ++j) {
      found = !used[j] && (ARegBases[j] == bB);
      if (found) {
        bIndices.push_back(j);
        used[j] = true;
        break;
      }
    }
    if (!found)
      return std::nullopt; // A basis from B not found in A.
  }
  // Append remaining indices from A (preserving their original order).
  SmallVector<size_t> remainingIndices;
  for (size_t i = 0; i < ARegBases.size(); ++i) {
    if (!used[i])
      remainingIndices.push_back(i);
  }
  SmallVector<size_t> permOrder = to_vector(llvm::concat<size_t>(
      left ? bIndices : remainingIndices, left ? remainingIndices : bIndices));
  return ColumnAction(permOrder, kReg, ARegBases.size());
}

ColumnAction actionRemoveBroadcastedRegs(const LinearLayout &layout) {
  assert(layout.getNumInDims() != 0);
  auto kReg = *layout.getInDimNames().begin();
  assert(kReg.str() == "register");

  // Drop the bases that are zero
  const auto &bases = layout.getBases().lookup(kReg);
  SmallVector<size_t> permOrder;
  for (size_t i = 0; i < bases.size(); ++i) {
    if (!llvm::all_of(bases[i], [](size_t x) { return x == 0; })) {
      permOrder.push_back(i);
    }
  }
  return ColumnAction(permOrder, kReg, bases.size(),
                      layout.getInputRadices(kReg));
}
std::pair<int64_t, ColumnAction>
actionAdditiveStrides(const LinearLayout &layout, const LinearLayout addrLayout,
                      uint64_t maskSpanOffsets, uint64_t maskSpanBlocks,
                      int64_t regsPerInst) {
  // General idea:
  // We want to swap an xor into an addition when computing the register
  // offsets and blocks. We can do this if each output component of this
  // register is disjoint from the corresponding components from
  // lanes/warps/blocks and any affine offset.
  //
  // Additionally, the lowering only ever evaluates register offsets at indices
  // that are multiples of `regsPerInst` (the inner-loop stride, matching one
  // vectorised instruction). The first `log2(regsPerInst)` register bases are
  // therefore never selected by any computed index and are trivially additive,
  // independent of their basis value. We force them into the "additive" group
  // unconditionally so that the returned `nAdditive` is always at least
  // `regsPerInst`. In particular, callers do not need to pre-zero those bases
  // for the invariant to hold.
  assert(layout.getNumInDims() != 0);
  if (regsPerInst <= 0 || !llvm::isPowerOf2_64(regsPerInst))
    llvm_unreachable("regsPerInst must be a positive power of two");
  auto kReg = *layout.getInDimNames().begin();
  assert(kReg.str() == "register");
  assert(layout.getNumOutDims() == 2);
  assert(layout.getOutDims() == addrLayout.getOutDims());
  const size_t regBasisPerVec = llvm::Log2_64(regsPerInst);
  uint64_t offsetBits = maskSpanOffsets;
  uint64_t blockBits = maskSpanBlocks;
  auto addrInDims = llvm::to_vector(addrLayout.getInDimNames());
  auto addrOutDims = llvm::to_vector(addrLayout.getOutDimNames());
  offsetBits |= getOutputBasisMask(addrLayout, addrInDims, addrOutDims[0]);
  blockBits |= getOutputBasisMask(addrLayout, addrInDims, addrOutDims[1]);
  SmallVector<size_t> front, back;
  auto layoutNamedBases = layout.getBases();
  const auto &regBases = layoutNamedBases.lookup(kReg);
  if (regBases.size() < regBasisPerVec)
    llvm_unreachable(
        "layout must have at least log2(regsPerInst) register bases");

  // If a register basis overlaps the dynamic address, it must remain in the
  // outer xor offset. Any other register basis that overlaps it must remain
  // there as well; otherwise the inner loop would add two non-disjoint values.
  // Compute this transitive closure before partitioning the register bases.
  SmallVector<bool> isAdditive(regBases.size(), true);
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto [idx, basis] : llvm::enumerate(regBases)) {
      if (idx < regBasisPerVec || !isAdditive[idx])
        continue;
      if ((basis[0] & offsetBits) == 0 && (basis[1] & blockBits) == 0)
        continue;
      isAdditive[idx] = false;
      offsetBits |= basis[0];
      blockBits |= basis[1];
      changed = true;
    }
  }

  for (size_t idx = 0; idx < regBases.size(); ++idx) {
    if (idx < regBasisPerVec || isAdditive[idx]) {
      front.push_back(idx);
    } else {
      back.push_back(idx);
    }
  }
  auto permOrder = to_vector(llvm::concat<size_t>(front, back));
  return {1 << front.size(),
          ColumnAction(permOrder, kReg, layout.getInDimSizeLog2(kReg))};
}

SmallVector<Value> broadcastAs(const SmallVector<Value> &values,
                               const LinearLayout &layout) {
  assert(layout.getNumInDims() != 0);
  auto kReg = *layout.getInDimNames().begin();
  assert(kReg.str() == "register");
  uint32_t broadcastMask = layout.getFreeVariableMasks().lookup(kReg);
  auto radices = layout.getInputRadices(kReg);
  if (llvm::any_of(radices, [](int32_t radix) { return radix != 2; })) {
    // A unique register is the mixed-radix number of the unmasked digits.
    SmallVector<Value> ret;
    for (int32_t i = 0; i < layout.getInDimSize(kReg); ++i) {
      int32_t rest = i, srcIdx = 0, weight = 1;
      for (auto [digit, radix] : llvm::enumerate(radices)) {
        if (!(broadcastMask & (1u << digit))) {
          srcIdx += rest % radix * weight;
          weight *= radix;
        }
        rest /= radix;
      }
      ret.push_back(values[srcIdx]);
    }
    return ret;
  }
  assert((layout.getInDimSize(kReg) / (1 << llvm::popcount(broadcastMask))) ==
         values.size());

  std::vector<std::vector<int32_t>> newBases;
  int i = 0;
  for (int j = 0; j < layout.getInDimSizeLog2(kReg); j++) {
    if (broadcastMask & (1 << j)) {
      newBases.push_back({0});
    } else {
      newBases.push_back({1 << i});
      i++;
    }
  }
  auto newLayout = LinearLayout({{kReg, std::move(newBases)}}, {kReg});
  SmallVector<Value> ret;

  ret.reserve(newLayout.getInDimSize(kReg));
  for (int i = 0; i < newLayout.getInDimSize(kReg); i++) {
    int32_t srcIdx = newLayout.apply({{kReg, i}}).begin()->second;
    ret.push_back(values[srcIdx]);
  }
  return ret;
}

// Compute the supremum of two lists.
// If the supremum is not unique, we return the first list first
// Error out if the supremum does not exist
// e.g. sup([a, b], [a, c]) = [a, b, c], sup([a, b], [b, c]) = [a, b, c]
//      sup([a, b], [b, a]) = error! Supremum does not exist.
SmallVector<StringAttr> supremum(const SmallVector<StringAttr> &x,
                                 const SmallVector<StringAttr> &y) {
  llvm::SetVector<StringAttr> result;
  DenseMap<StringAttr, int> posX, posY;
  for (auto [idx, elem] : llvm::enumerate(x))
    posX[elem] = idx;
  for (auto [idx, elem] : llvm::enumerate(y))
    posY[elem] = idx;
  int i = 0, j = 0;
  const int INF = std::numeric_limits<int>::max();
  while (i < x.size() || j < y.size()) {
    while (i < x.size() && result.contains(x[i]))
      ++i;
    while (j < y.size() && result.contains(y[j]))
      ++j;
    if (i >= x.size() && j >= y.size())
      break;
    if (i < x.size() && j < y.size() && x[i] == y[j]) {
      if (posY[x[i]] < j)
        llvm_unreachable("Supremum does not exist");
      result.insert(x[i]);
      ++i, ++j;
      continue;
    }
    int candX = INF, candY = INF;
    if (i < x.size()) {
      if (posY.count(x[i]) && posY[x[i]] >= j)
        candX = posY[x[i]];
    }
    if (j < y.size()) {
      if (posX.count(y[j]) && posX[y[j]] >= i)
        candY = posX[y[j]];
    }
    if (i < x.size() && candX == INF) {
      result.insert(x[i]);
      ++i;
      continue;
    }
    if (j < y.size() && candY == INF) {
      result.insert(y[j]);
      ++j;
      continue;
    }
    if (candX <= candY) {
      if (posY[x[i]] < j)
        llvm_unreachable("Supremum does not exist");
      result.insert(x[i]);
      ++i;
    } else {
      if (posX[y[j]] < i)
        llvm_unreachable("Supremum does not exist");
      result.insert(y[j]);
      ++j;
    }
  }
  return to_vector(result);
}

LinearLayout reshapeLayout(MLIRContext *ctx, LinearLayout layout,
                           ArrayRef<int64_t> shape) {
  // reshapeOuts needs pow2 out dims.
  if (layout.hasNonPow2OutDim()) {
    std::optional<LinearLayout> reshaped = tryReshapeLayout(ctx, layout, shape);
    assert(reshaped && "the reshape splits a non-pow2 index digit");
    return *reshaped;
  }
  int rank = shape.size();
  auto srcOutDims = to_vector(layout.getOutDimNames());
  std::reverse(srcOutDims.begin(), srcOutDims.end());
  auto newOutDims = standardOutDimPairs(ctx, shape);
  std::reverse(newOutDims.begin(), newOutDims.end());
  return layout.transposeOuts(srcOutDims)
      .reshapeOuts(newOutDims)
      .transposeOuts(standardOutDimNames(ctx, rank));
}

namespace {
// The digits of a row-major index into `shape`, least significant first: each
// dim's binary digits, then one odd digit if its size is not a power of two.
struct IndexDigits {
  SmallVector<int32_t> radices;
  SmallVector<int32_t> dimOf;
  SmallVector<int32_t> firstOf;
};

IndexDigits getIndexDigits(ArrayRef<int64_t> shape) {
  IndexDigits digits;
  digits.firstOf.resize(shape.size());
  for (int dim = shape.size() - 1; dim >= 0; --dim) {
    digits.firstOf[dim] = digits.radices.size();
    int32_t size = shape[dim];
    int32_t twoPart = LinearLayout::twoPartOf(size);
    digits.radices.append(llvm::Log2_32(twoPart), 2);
    if (size != twoPart)
      digits.radices.push_back(size / twoPart);
    digits.dimOf.resize(digits.radices.size(), dim);
  }
  return digits;
}
} // namespace

std::optional<LinearLayout> tryReshapeLayout(MLIRContext *ctx,
                                             const LinearLayout &layout,
                                             ArrayRef<int64_t> shape) {
  auto srcDims = to_vector(layout.getOutDimNames());
  SmallVector<int64_t> srcShape;
  for (StringAttr dim : srcDims)
    srcShape.push_back(layout.getOutDimSize(dim));
  IndexDigits src = getIndexDigits(srcShape);
  IndexDigits dst = getIndexDigits(shape);
  if (src.radices != dst.radices)
    return std::nullopt;
  // Each coefficient keeps its digit; an odd digit sits above its new dim's
  // binary digits, so its coefficient is scaled by their weight.
  LinearLayout::BasesT bases;
  for (const auto &[inDim, inDimBases] : layout.getBases()) {
    auto &newBases = bases[inDim];
    for (int32_t digit = 0, e = inDimBases.size(); digit < e; ++digit) {
      std::vector<int32_t> basis(shape.size(), 0);
      for (auto [i, outDim] : llvm::enumerate(srcDims)) {
        auto [binary, odd] = layout.getBasisContribution(inDim, digit, outDim);
        int32_t numBinary = llvm::Log2_32(
            LinearLayout::twoPartOf(layout.getOutDimSize(outDim)));
        for (int32_t pos = 0; pos <= numBinary; ++pos) {
          int32_t coeff = pos < numBinary ? (binary >> pos) & 1 : odd;
          if (coeff == 0)
            continue;
          int32_t flat = src.firstOf[i] + pos;
          int32_t dim = dst.dimOf[flat];
          basis[dim] += coeff << (flat - dst.firstOf[dim]);
        }
      }
      newBases.push_back(std::move(basis));
    }
  }
  return LinearLayout(std::move(bases), layout.getInputRadices(),
                      standardOutDimPairs(ctx, shape), layout.isSurjective());
}

LinearLayout transposeLinearLayout(LinearLayout layout, ArrayRef<int> order) {
  // Transpose the tile layout.
  auto namedBases = layout.getBases();
  // move the most outer dimensions to the inner most position.

  for (auto &bases : llvm::make_second_range(namedBases)) {
    for (auto &b : bases) {
      std::vector<int32_t> newB;
      for (auto i : order) {
        newB.push_back(b[i]);
      }
      b = std::move(newB);
    }
  }
  if (!layout.isRadix2()) {
    // NPOT sizes are not inferable from bases: each dim takes its source's.
    auto names = to_vector(layout.getOutDimNames());
    SmallVector<std::pair<StringAttr, int32_t>> outDims;
    for (auto [name, src] : llvm::zip_equal(names, order))
      outDims.push_back({name, layout.getOutDimSize(names[src])});
    return LinearLayout(std::move(namedBases), layout.getInputRadices(),
                        outDims, /*requireSurjective=*/false);
  }
  return LinearLayout(std::move(namedBases),
                      to_vector(layout.getOutDimNames()));
}

std::pair<int, ColumnAction>
largestVectorisation(MLIRContext *ctx, const LinearLayout &cvt, int bitwidth,
                     std::optional<int> maybeMaxVecElems) {
  // Find the largest vectorisation we can use:
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  StringAttr kReg = S("register");
  StringAttr kOffset = S("offset");
  LinearLayout quot;
  LinearLayout tile;
  ColumnAction permutation;
  // If there are restrictions on the vectorisation, we don't allow
  // permutations.
  auto allowPerm = !maybeMaxVecElems.has_value();
  auto maxVecElems = maybeMaxVecElems.value_or(128 / bitwidth);
  for (int v = maxVecElems; v >= 1; v /= 2) {
    tile = LinearLayout::identity1D(v, kReg, kOffset);
    auto maybePerm = regPermForDivide(cvt, tile, /*left=*/true);
    if (!maybePerm) {
      continue;
    }
    permutation = *maybePerm;
    if (!allowPerm && !permutation.isIdentity()) {
      continue;
    }
    auto newCvt = permutation.apply(cvt);
    auto maybeQuot = divideLeft(newCvt, tile);
    if (!maybeQuot) {
      continue;
    }
    return {v, permutation};
  }
  llvm_unreachable("Vectorization < 1 is not valid");
}

std::optional<LinearLayout> getReps(const LinearLayout &cvt,
                                    const LinearLayout &tile) {

  // Ensure tile out-dims are subset of cvt out-dims.
  for (auto od : tile.getOutDimNames())
    assert(cvt.hasOutDim(od) && "tile out-dims must be contained in cvt");

  // Build a per-out-dimension mask by OR-ing all tile bases that touch it.
  llvm::SmallDenseMap<StringAttr, int32_t> tileMaskPerOutDim;
  auto tileInDims = llvm::to_vector(tile.getInDimNames());
  for (StringAttr od : cvt.getOutDimNames())
    tileMaskPerOutDim[od] =
        tile.hasOutDim(od) ? getOutputBasisMask(tile, tileInDims, od) : 0;

  // Build reps with the same in/out dims as cvt, but zeroing out the leading
  // inB bases (per in-dim) and keeping the remainder bases unchanged from cvt.
  LinearLayout::BasesT repsBases;
  for (StringAttr id : cvt.getInDimNames()) {
    int inA = cvt.getInDimSizeLog2(id);
    int inB = tile.hasInDim(id) ? tile.getInDimSizeLog2(id) : 0;
    if (inB > inA) {
      return std::nullopt;
    }

    std::vector<std::vector<int32_t>> basesForDim;
    basesForDim.reserve(inA);

    // 1) Validate the starting bases match exactly.
    for (int i = 0; i < inB; ++i) {
      for (StringAttr od : cvt.getOutDimNames()) {
        int a = cvt.getBasis(id, i, od);
        int b = tile.hasOutDim(od) ? tile.getBasis(id, i, od) : 0;
        if (a != b) {
          return std::nullopt;
        }
      }
    }

    // 2) Validate no overlap: the remaining cvt bases must have zeros in all
    //    tile-bit positions (computed as OR of all tile bases) for each
    //    out-dim.
    for (int i = inB; i < inA; ++i) {
      for (StringAttr od : cvt.getOutDimNames()) {
        int32_t mask = tileMaskPerOutDim.lookup(od);
        if (mask == 0)
          continue;
        int v = cvt.getBasis(id, i, od);
        if ((v & mask) != 0) {
          return std::nullopt;
        }
      }
    }

    // 3) Emit reps bases: first inB as all-zeros; remainder copied from cvt.
    for (int i = 0; i < inB; ++i) {
      std::vector<int32_t> zero(cvt.getNumOutDims(), 0);
      basesForDim.push_back(std::move(zero));
    }
    for (int i = inB; i < inA; ++i) {
      std::vector<int32_t> keep;
      keep.reserve(cvt.getNumOutDims());
      for (StringAttr od : cvt.getOutDimNames())
        keep.push_back(cvt.getBasis(id, i, od));
      basesForDim.push_back(std::move(keep));
    }

    repsBases[id] = std::move(basesForDim);
  }

  return LinearLayout(std::move(repsBases), cvt.getOutDims(),
                      /*requireSurjective=*/false);
}

LinearLayout removeStandardDim(const LinearLayout &layout, int dim) {
  auto rank = layout.getNumOutDims();
  assert(rank > 0);
  assert(dim < rank);
  auto *ctx = layout.getOutDimNames().begin()->getContext();
  auto dims = to_vector(layout.getOutDimNames());
  assert(dims == standardOutDimNames(ctx, rank));
  dims.erase(dims.begin() + dim);
  auto newLayout = layout.sublayout(to_vector(layout.getInDimNames()), dims);
  auto newDims = standardOutDimNames(ctx, rank - 1);
  SmallVector<std::pair<StringAttr, StringAttr>> renames;
  for (auto [oldDim, newDim] : llvm::zip_equal(dims, newDims)) {
    renames.push_back({oldDim, newDim});
  }
  return renameLinearLayoutDims(newLayout, {}, renames);
}

} // namespace mlir::triton
