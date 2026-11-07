#include "triton/Tools/LinearLayout.h"

#include <map>
#include <numeric>

#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

// The LinearLayout bodies for input digits whose radix is not 2 and out dims
// whose size is not a power of two.

namespace mlir::triton {

namespace {
// Enumerations and tables stay within this many entries.
constexpr int64_t kMaxValues = int64_t{1} << 20;

// Calls fn(ins, index) for every input, counting with the first digit lowest.
template <typename Fn>
void forEachInput(const LinearLayout::RadicesT &radices, Fn fn) {
  SmallVector<std::pair<StringAttr, int32_t>> ins;
  SmallVector<int64_t> sizes;
  for (const auto &[inDim, dimRadices] : radices) {
    ins.push_back({inDim, 0});
    sizes.push_back(std::accumulate(dimRadices.begin(), dimRadices.end(),
                                    int64_t{1}, std::multiplies<>()));
  }
  for (int64_t index = 0;; ++index) {
    fn(ArrayRef(ins), index);
    size_t d = 0;
    for (; d < ins.size() && ++ins[d].second == sizes[d]; ++d)
      ins[d].second = 0;
    if (d == ins.size())
      return;
  }
}

// The landed odd-part encoding: one unit digit d adds twoPart*(d*c % oddPart).
bool isOddPartTable(int32_t size, ArrayRef<int32_t> offsets) {
  if (size <= 0)
    return false;
  int32_t twoPart = size & -size, oddPart = size / twoPart;
  if (oddPart < 2 || offsets.size() != size_t(oddPart))
    return false;
  int32_t coefficient = offsets[1] / twoPart;
  return std::gcd(coefficient, oddPart) == 1 &&
         llvm::all_of(llvm::enumerate(offsets), [&](auto entry) {
           return entry.value() ==
                  twoPart * (int64_t(entry.index()) * coefficient % oddPart);
         });
}

// Splits `values` over digits of `radices` (first digit low) into XORed
// binary digits plus a table over `table`. The binary digits are the radix-2
// digits whose unit value is below the largest 2^k that every nonzero table
// entry reaches, so a digit is promoted only when it can't be binary.
void splitValues(ArrayRef<int32_t> radices, ArrayRef<int32_t> values,
                 int32_t size, SmallVectorImpl<int32_t> &unit,
                 SmallVectorImpl<int32_t> &table,
                 std::vector<int32_t> &offsets) {
  int32_t numDigits = radices.size();
  SmallVector<int64_t> weights(numDigits + 1, 1);
  for (int32_t d = 0; d < numDigits; ++d)
    weights[d + 1] = weights[d] * radices[d];
  unit.assign(numDigits, 0);
  for (int32_t d = 0; d < numDigits; ++d)
    unit[d] = values[weights[d]];
  SmallVector<bool> binary(numDigits, false);
  SmallVector<int32_t> digits(numDigits, 0);
  for (int32_t k = llvm::countr_zero(static_cast<uint32_t>(size)); k >= 0;
       --k) {
    int64_t bound = int64_t{1} << k;
    for (int32_t d = 0; d < numDigits; ++d)
      binary[d] = radices[d] == 2 && unit[d] != 0 && unit[d] < bound;
    bool ok = true;
    llvm::fill(digits, 0);
    for (int64_t i = 0; ok && i < weights[numDigits]; ++i) {
      int32_t xorPart = 0;
      int64_t rest = 0;
      for (int32_t d = 0; d < numDigits; ++d) {
        if (binary[d])
          xorPart ^= digits[d] ? unit[d] : 0;
        else
          rest += digits[d] * weights[d];
      }
      int32_t entry = values[rest];
      ok = (entry == 0 || entry >= bound) &&
           values[i] == int64_t{xorPart} + entry;
      for (int32_t d = 0; d < numDigits && ++digits[d] == radices[d]; ++d)
        digits[d] = 0;
    }
    if (ok)
      break;
  }
  // A non-binary digit joins the table only if some entry depends on it.
  SmallVector<bool> used(numDigits, false);
  llvm::fill(digits, 0);
  for (int64_t i = 0; i < weights[numDigits]; ++i) {
    bool onTable =
        llvm::none_of(llvm::seq<int32_t>(0, numDigits),
                      [&](int32_t d) { return binary[d] && digits[d] != 0; });
    for (int32_t d = 0; d < numDigits && onTable; ++d)
      if (digits[d] != 0 && values[i] != values[i - digits[d] * weights[d]])
        used[d] = true;
    for (int32_t d = 0; d < numDigits && ++digits[d] == radices[d]; ++d)
      digits[d] = 0;
  }
  table.clear();
  int64_t entries = 1;
  for (int32_t d = 0; d < numDigits; ++d) {
    if (!binary[d] && used[d]) {
      table.push_back(d);
      entries *= radices[d];
    }
  }
  offsets.assign(entries, 0);
  for (int64_t t = 0; t < entries; ++t) {
    int64_t quotient = t, index = 0;
    for (int32_t d : table) {
      index += quotient % radices[d] * weights[d];
      quotient /= radices[d];
    }
    offsets[t] = values[index];
  }
}
} // namespace

/*static*/ int32_t LinearLayout::twoPartOf(int32_t size) {
  return int32_t{1} << llvm::countr_zero(static_cast<uint32_t>(size));
}

bool LinearLayout::isOddPartOutDim(StringAttr outDim) const {
  int32_t size = getOutDimSize(outDim), twoPart = twoPartOf(size);
  int32_t outIndex = getOutDimIndex(outDim);
  ArrayRef<std::pair<StringAttr, int32_t>> digits = getTableDigits(outDim);
  for (const auto &[inDim, inDimBases] : bases)
    for (auto [pos, basis] : llvm::enumerate(inDimBases))
      if (basis[outIndex] >= twoPart &&
          !llvm::is_contained(digits, std::pair(inDim, int32_t(pos))))
        return false;
  return digits.empty() ||
         (digits.size() == 1 &&
          getInputRadix(digits[0].first, digits[0].second) == size / twoPart &&
          isOddPartTable(size, getTableOffsets(outDim)));
}

bool LinearLayout::hasOddPartForm() const {
  return llvm::all_of(getOutDimNames(), [&](StringAttr outDim) {
    return isOddPartOutDim(outDim);
  });
}

LinearLayout::BasisContribution
LinearLayout::getBasisContribution(StringAttr inDim, int32_t pos,
                                   StringAttr outDim) const {
  int32_t basis = getBasis(inDim, pos, outDim);
  if (!isOddPartOutDim(outDim))
    return {basis, -1};
  int32_t twoPart = twoPartOf(getOutDimSize(outDim));
  return {basis % twoPart, basis / twoPart};
}

LinearLayout LinearLayout::binaryPart() const {
  SmallVector<std::pair<StringAttr, int32_t>> partOutDims;
  for (auto [outDim, size] : outDims)
    partOutDims.push_back({outDim, twoPartOf(size)});
  BasesT partBases;
  for (const auto &[inDim, inDimBases] : bases) {
    for (auto [pos, basis] : llvm::enumerate(inDimBases)) {
      std::vector<int32_t> partBasis(outDims.size(), 0);
      for (auto [outIndex, outDim] : llvm::enumerate(getOutDimNames())) {
        if (llvm::is_contained(getTableDigits(outDim),
                               std::pair(inDim, int32_t(pos))))
          continue;
        partBasis[outIndex] = basis[outIndex];
        int32_t &size = partOutDims[outIndex].second;
        size = std::max<int32_t>(size, llvm::NextPowerOf2(basis[outIndex]));
      }
      partBases[inDim].push_back(std::move(partBasis));
    }
  }
  return LinearLayout(std::move(partBases), partOutDims,
                      /*requireSurjective=*/false);
}

int32_t LinearLayout::getMixedRadixCodomainRank() const {
  int32_t result = 0;
  for (auto [outDim, size] : outDims) {
    if (isMixedRadixOutDim(outDim))
      result += llvm::countr_zero(static_cast<uint32_t>(size)) + 1;
    else
      result += llvm::Log2_32(size);
  }
  return result;
}

int32_t LinearLayout::getMixedRadixRank() const {
  int32_t result = binaryPart().rank;
  bool general = false;
  for (const auto &[outDim, digits] : tableDigits) {
    if (isOddPartOutDim(outDim))
      ++result;
    else
      general = true;
  }
  if (!general)
    return result;
  int32_t codomain = getMixedRadixCodomainRank();
  int64_t totalOut = 1;
  for (auto [outDim, size] : outDims)
    totalOut = std::min<int64_t>(totalOut * size, int64_t{1} << 40);
  int64_t totalIn = getTotalInDimSize();
  if (totalIn < totalOut || totalOut > kMaxValues * 16 || totalIn > kMaxValues)
    return std::min(result, codomain - 1);
  std::vector<bool> seen(totalOut, false);
  int64_t numSeen = 0;
  forEachInput(inputRadices, [&](auto ins, int64_t) {
    int64_t flat = 0, stride = 1;
    for (auto [outDimAndSize, out] : llvm::zip(outDims, apply(ins))) {
      flat += out.second * stride;
      stride *= outDimAndSize.second;
    }
    numSeen += !seen[flat];
    seen[flat] = true;
  });
  return numSeen == totalOut ? codomain : std::min(result, codomain - 1);
}

std::optional<std::string> LinearLayout::checkInputRadices() const {
  if (!llvm::equal(getInDimNames(), llvm::make_first_range(inputRadices)) ||
      llvm::any_of(bases, [&](const auto &dimAndBases) {
        return dimAndBases.second.size() !=
               inputRadices.lookup(dimAndBases.first).size();
      }))
    return "invalid input radices: expected one per basis, in the order of "
           "the bases:" +
           toString();
  int64_t totalSize = 1;
  for (const auto &[inDim, radices] : inputRadices) {
    for (auto [radix, basis] : llvm::zip(radices, bases.find(inDim)->second)) {
      if (radix < 2)
        return "invalid input radix " + std::to_string(radix) + " of in-dim '" +
               inDim.str() + "'; radices must be at least 2";
      if (radix != 2 &&
          llvm::any_of(llvm::zip(basis, getOutDimNames()), [&](auto entry) {
            return std::get<0>(entry) != 0 &&
                   !isMixedRadixOutDim(std::get<1>(entry));
          }))
        return "invalid input digit of radix " + std::to_string(radix) +
               " in in-dim '" + inDim.str() +
               "': its basis must be zero on Binary outputs:" + toString();
      totalSize = std::min<int64_t>(totalSize * radix, int64_t{1} << 32);
    }
  }
  if (!isRadix2() && totalSize > std::numeric_limits<int32_t>::max())
    return "invalid total input size " + std::to_string(totalSize) +
           "; must be at most INT32_MAX";
  return std::nullopt;
}

std::optional<std::string> LinearLayout::checkTables() {
  for (const auto &[outDim, digits] : tableDigits) {
    if (!hasOutDim(outDim) || !isMixedRadixOutDim(outDim))
      return "invalid offset table: out-dim '" + outDim.str() +
             "' is not a mixed-radix output";
    int64_t entries = 1;
    for (auto [i, digit] : llvm::enumerate(digits)) {
      auto [inDim, pos] = digit;
      if (!hasInDim(inDim) || pos < 0 || pos >= getInDimSizeLog2(inDim) ||
          llvm::is_contained(ArrayRef(digits).take_front(i), digit))
        return "invalid table digit " + std::to_string(pos) + " of in-dim '" +
               inDim.str() + "' for out-dim '" + outDim.str() + "'";
      entries *= getInputRadix(inDim, pos);
      if (entries > kMaxValues)
        return "invalid offset table: out-dim '" + outDim.str() +
               "' has more than 2^20 entries";
    }
    size_t numOffsets = tableOffsets.lookup(outDim).size();
    if (numOffsets != size_t(entries))
      return "invalid offset table: " + std::to_string(numOffsets) +
             " offsets for " + std::to_string(entries) +
             " table indices of out-dim '" + outDim.str() + "'";
  }
  if (llvm::any_of(tableOffsets, [&](const auto &entry) {
        return !tableDigits.contains(entry.first);
      }))
    return "invalid offset table: offsets without table digits";

  // An odd-part table is the landed encoding, whose digit's basis carries it.
  // The others are checked by value below.
  BasesT newBases = bases;
  decltype(tableDigits) given;
  for (const auto &[outDim, digits] : tableDigits) {
    if (!isOddPartOutDim(outDim)) {
      given[outDim] = digits;
      continue;
    }
    newBases[digits[0].first][digits[0].second][getOutDimIndex(outDim)] =
        getTableOffsets(outDim)[1];
  }
  // Ops on a general table enumerate the inputs, so they stay within the cap.
  int64_t totalIn = 1;
  for (const auto &[inDim, radices] : inputRadices)
    for (int32_t radix : radices)
      totalIn = std::min(totalIn * radix, kMaxValues + 1);
  if (!given.empty() && totalIn > kMaxValues)
    return "invalid offset table: a general table allows at most 2^20 "
           "inputs:" +
           toString();

  // Out dims without a table take the landed odd-digit form: radix-2 digits
  // touch only binary parts, and each odd part has at most one non-binary
  // digit, which permutes it.
  SmallVector<StringAttr> outDimNames = llvm::to_vector(getOutDimNames());
  SmallVector<bool> hasOddOwner(outDims.size(), false);
  for (const auto &[inDim, inDimBases] : newBases) {
    for (int32_t digit = 0, e = inDimBases.size(); digit < e; ++digit) {
      int32_t radix = getInputRadix(inDim, digit);
      bool ownsOutput = false;
      for (auto [outIndex, outDim] : llvm::enumerate(outDimNames)) {
        if (given.contains(outDim))
          continue;
        int32_t twoPart = twoPartOf(getOutDimSize(outDim));
        int32_t binary = inDimBases[digit][outIndex] % twoPart;
        int32_t odd = inDimBases[digit][outIndex] / twoPart;
        if (radix == 2) {
          if (odd != 0)
            return "invalid radix-2 digit: it has an odd mixed-radix "
                   "contribution:" +
                   toString();
          continue;
        }
        if (binary != 0)
          return "invalid non-binary digit: it has a binary mixed-radix "
                 "contribution:" +
                 toString();
        if (odd == 0)
          continue;
        int32_t oddPart = getOutDimSize(outDim) / twoPart;
        if (ownsOutput || hasOddOwner[outIndex] || radix != oddPart ||
            std::gcd(odd, oddPart) != 1)
          return "NPOT layout not yet supported: each odd part needs one "
                 "non-binary digit of that radix with a unit coefficient:" +
                 toString();
        hasOddOwner[outIndex] = true;
        ownsOutput = true;
      }
    }
  }

  // Build the tables in locals, so a failure leaves this layout unchanged.
  decltype(tableDigits) newDigits;
  decltype(tableOffsets) newOffsets;
  for (auto [outIndex, outDimAndSize] : llvm::enumerate(outDims)) {
    auto [outDim, size] = outDimAndSize;
    if (!isMixedRadixOutDim(outDim))
      continue;
    int32_t twoPart = twoPartOf(size), oddPart = size / twoPart;
    auto givenDigits = given.find(outDim);
    if (givenDigits == given.end()) {
      for (const auto &[inDim, inDimBases] : newBases) {
        for (auto [pos, basis] : llvm::enumerate(inDimBases)) {
          if (getInputRadix(inDim, pos) == 2 || basis[outIndex] == 0)
            continue;
          int64_t coefficient = basis[outIndex] / twoPart;
          newDigits[outDim] = {{inDim, int32_t(pos)}};
          auto &offsets = newOffsets[outDim];
          for (int64_t i = 0; i < oddPart; ++i)
            offsets.push_back(twoPart * (i * coefficient % oddPart));
        }
      }
      continue;
    }
    // The digits reaching outDim, with each table digit's weight in the
    // given index.
    ArrayRef<int32_t> givenOffsets = tableOffsets.find(outDim)->second;
    SmallVector<std::pair<StringAttr, int32_t>> digits;
    SmallVector<int32_t> radices;
    SmallVector<int64_t> givenWeights;
    for (const auto &[inDim, inDimBases] : bases) {
      for (auto [pos, basis] : llvm::enumerate(inDimBases)) {
        std::pair<StringAttr, int32_t> digit(inDim, pos);
        int32_t radix = getInputRadix(inDim, pos);
        const auto *it = llvm::find(givenDigits->second, digit);
        int64_t weight = 0;
        if (it == givenDigits->second.end()) {
          if (basis[outIndex] == 0)
            continue;
          if (radix != 2)
            return "invalid non-binary digit: it reaches out-dim '" +
                   outDim.str() + "' outside its table:" + toString();
        } else {
          weight = 1;
          for (const auto &prev : ArrayRef(givenDigits->second.begin(), it))
            weight *= getInputRadix(prev.first, prev.second);
        }
        digits.push_back(digit);
        radices.push_back(radix);
        givenWeights.push_back(weight);
      }
    }
    int64_t numValues = 1;
    for (int32_t radix : radices) {
      numValues *= radix;
      if (numValues > kMaxValues)
        return "invalid offset table: out-dim '" + outDim.str() +
               "' has more than 2^20 values to check";
    }
    std::vector<int32_t> values(numValues);
    SmallVector<int32_t> digitValues(digits.size(), 0);
    for (int64_t i = 0; i < numValues; ++i) {
      int32_t xorPart = 0;
      int64_t index = 0;
      for (auto [d, digit] : llvm::enumerate(digits)) {
        if (givenWeights[d] != 0)
          index += digitValues[d] * givenWeights[d];
        else if (digitValues[d] != 0)
          xorPart ^= bases.find(digit.first)->second[digit.second][outIndex];
      }
      int64_t value = int64_t{xorPart} + givenOffsets[index];
      if (value < 0 || value >= size || (i == 0 && value != 0))
        return "invalid offset table: out-dim '" + outDim.str() + "' reaches " +
               std::to_string(value) + " at input " + std::to_string(i) +
               "; values must be in [0, " + std::to_string(size) +
               ") with 0 at input 0";
      values[i] = value;
      for (size_t d = 0; d < digits.size() && ++digitValues[d] == radices[d];
           ++d)
        digitValues[d] = 0;
    }
    SmallVector<int32_t> unit, table;
    std::vector<int32_t> offsets;
    splitValues(radices, values, size, unit, table, offsets);
    for (auto [d, digit] : llvm::enumerate(digits))
      newBases[digit.first][digit.second][outIndex] = unit[d];
    if (table.empty())
      continue;
    for (int32_t d : table)
      newDigits[outDim].push_back(digits[d]);
    newOffsets[outDim] = std::move(offsets);
  }
  bases = std::move(newBases);
  tableDigits = std::move(newDigits);
  tableOffsets = std::move(newOffsets);
  return std::nullopt;
}

/*static*/ LinearLayout LinearLayout::mixedRadixStrided1D(int32_t size,
                                                          int32_t stride,
                                                          StringAttr inDim,
                                                          StringAttr outDim) {
  int32_t twoPart = twoPartOf(size);
  BasesT bases;
  RadicesT radices;
  for (int32_t i = 1; i < twoPart; i *= 2) {
    bases[inDim].push_back({i * stride});
    radices[inDim].push_back(2);
  }
  bases[inDim].push_back({twoPart * stride});
  radices[inDim].push_back(size / twoPart);
  return LinearLayout(std::move(bases), std::move(radices),
                      {{outDim, stride * size}},
                      /*requireSurjective=*/stride == 1);
}

/*static*/ std::optional<LinearLayout> LinearLayout::fromValues(
    RadicesT radices, ArrayRef<std::pair<StringAttr, int32_t>> outDims,
    ArrayRef<std::vector<int32_t>> values, std::string *diagnostic) {
  SmallVector<std::pair<StringAttr, int32_t>> digits;
  SmallVector<int32_t> digitRadices;
  BasesT bases;
  for (const auto &[inDim, dimRadices] : radices) {
    bases[inDim].assign(dimRadices.size(),
                        std::vector<int32_t>(outDims.size(), 0));
    for (auto [pos, radix] : llvm::enumerate(dimRadices)) {
      digits.push_back({inDim, int32_t(pos)});
      digitRadices.push_back(radix);
    }
  }
  decltype(tableDigits) newDigits;
  decltype(tableOffsets) newOffsets;
  for (auto [outIndex, outDimAndSize] : llvm::enumerate(outDims)) {
    auto [outDim, size] = outDimAndSize;
    SmallVector<int32_t> unit, table;
    std::vector<int32_t> offsets;
    splitValues(digitRadices, values[outIndex], size, unit, table, offsets);
    if (!table.empty() && llvm::isPowerOf2_32(size)) {
      if (diagnostic)
        *diagnostic = "out-dim '" + outDim.str() + "' of power-of-two size " +
                      std::to_string(size) +
                      " would need an offset table; its values are not "
                      "linear over GF(2)";
      return std::nullopt;
    }
    for (auto [d, digit] : llvm::enumerate(digits))
      bases[digit.first][digit.second][outIndex] = unit[d];
    for (int32_t d : table)
      newDigits[outDim].push_back(digits[d]);
    if (!table.empty())
      newOffsets[outDim] = std::move(offsets);
  }
  return tryCreate(std::move(bases), std::move(radices), outDims,
                   std::move(newDigits), std::move(newOffsets),
                   /*requireSurjective=*/false, diagnostic);
}

/*static*/ std::optional<LinearLayout> LinearLayout::fromFunction(
    RadicesT radices, ArrayRef<std::pair<StringAttr, int32_t>> outDims,
    function_ref<void(ArrayRef<std::pair<StringAttr, int32_t>>,
                      MutableArrayRef<int32_t>)>
        fn,
    std::string *diagnostic) {
  int64_t numValues = 1;
  for (const auto &[inDim, dimRadices] : radices)
    for (int32_t radix : dimRadices)
      numValues = std::min(numValues * radix, kMaxValues + 1);
  if (numValues > kMaxValues) {
    if (diagnostic)
      *diagnostic = "more than 2^20 inputs to enumerate";
    return std::nullopt;
  }
  std::vector<std::vector<int32_t>> values(outDims.size(),
                                           std::vector<int32_t>(numValues));
  SmallVector<int32_t> outs(outDims.size(), 0);
  forEachInput(radices, [&](auto ins, int64_t index) {
    fn(ins, outs);
    for (auto [outIndex, out] : llvm::enumerate(outs))
      values[outIndex][index] = out;
  });
  return fromValues(std::move(radices), outDims, values, diagnostic);
}

/*static*/ std::optional<LinearLayout> LinearLayout::mixedRadixProduct(
    const LinearLayout &inner, const LinearLayout &outer,
    ArrayRef<StringAttr> inDims, ArrayRef<StringAttr> outDims) {
  // The odd-part rule of operator*: outer values land above inner's binary
  // part, and inner's quotient by its two-part moves above them.
  RadicesT radices;
  for (StringAttr inDim : inDims)
    for (const LinearLayout *layout : {&inner, &outer})
      if (layout->hasInDim(inDim))
        llvm::append_range(radices[inDim], layout->getInputRadices(inDim));
  auto sizeOn = [](const LinearLayout &layout, StringAttr outDim) {
    return layout.hasOutDim(outDim) ? layout.getOutDimSize(outDim) : 1;
  };
  auto valueOn = [](ArrayRef<std::pair<StringAttr, int32_t>> outs,
                    StringAttr outDim) -> int64_t {
    const auto *it =
        llvm::find_if(outs, [&](auto o) { return o.first == outDim; });
    return it == outs.end() ? 0 : it->second;
  };
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (StringAttr outDim : outDims)
    newOutDims.push_back(
        {outDim, sizeOn(inner, outDim) * sizeOn(outer, outDim)});
  return fromFunction(std::move(radices), newOutDims, [&](auto ins, auto outs) {
    SmallVector<std::pair<StringAttr, int32_t>> innerIns, outerIns;
    for (auto [inDim, value] : ins) {
      int32_t innerSize = inner.hasInDim(inDim) ? inner.getInDimSize(inDim) : 1;
      if (inner.hasInDim(inDim))
        innerIns.push_back({inDim, value % innerSize});
      if (outer.hasInDim(inDim))
        outerIns.push_back({inDim, value / innerSize});
    }
    auto innerOuts = inner.apply(innerIns), outerOuts = outer.apply(outerIns);
    for (auto [outIndex, outDim] : llvm::enumerate(outDims)) {
      int64_t twoPart = twoPartOf(sizeOn(inner, outDim));
      int64_t v = valueOn(innerOuts, outDim);
      outs[outIndex] =
          v % twoPart + twoPart * (valueOn(outerOuts, outDim) +
                                   sizeOn(outer, outDim) * (v / twoPart));
    }
  });
}

std::optional<LinearLayout> LinearLayout::reshapeMixedRadixOuts(
    ArrayRef<std::pair<StringAttr, int32_t>> newOutDims) const {
  // Flatten the outputs, first out-dim lowest, then split by the new sizes.
  int64_t newTotal = 1;
  for (auto [outDim, size] : newOutDims)
    newTotal *= size;
  if (newTotal != getTotalOutDimSize())
    return std::nullopt;
  return fromFunction(inputRadices, newOutDims, [&](auto ins, auto outs) {
    int64_t flat = 0, stride = 1;
    for (auto [outDimAndSize, out] : llvm::zip(outDims, apply(ins))) {
      flat += out.second * stride;
      stride *= outDimAndSize.second;
    }
    for (auto [outIndex, outDimAndSize] : llvm::enumerate(newOutDims)) {
      outs[outIndex] = flat % outDimAndSize.second;
      flat /= outDimAndSize.second;
    }
  });
}

std::optional<LinearLayout>
LinearLayout::concatMixedRadixIns(const LinearLayout &other) const {
  // Binary parts XOR and quotients by the two-part add, as for odd-part
  // layouts.
  RadicesT radices = inputRadices;
  for (const auto &entry : other.inputRadices)
    radices.insert(entry);
  int32_t numLhs = getNumInDims();
  return fromFunction(
      std::move(radices), llvm::to_vector(outDims), [&](auto ins, auto outs) {
        auto lhs = apply(ins.take_front(numLhs));
        auto rhs = other.apply(ins.drop_front(numLhs));
        for (auto [outIndex, outDimAndSize] : llvm::enumerate(outDims)) {
          int32_t size = outDimAndSize.second, twoPart = twoPartOf(size);
          int32_t a = lhs[outIndex].second, b = rhs[outIndex].second;
          outs[outIndex] =
              (a % twoPart ^ b % twoPart) +
              twoPart * ((a / twoPart + b / twoPart) % (size / twoPart));
        }
      });
}

std::optional<LinearLayout>
LinearLayout::concatMixedRadixOuts(const LinearLayout &other) const {
  auto newOutDims = llvm::to_vector(outDims);
  llvm::append_range(newOutDims, other.outDims);
  return fromFunction(inputRadices, newOutDims, [&](auto ins, auto outs) {
    auto lhs = apply(ins), rhs = other.apply(ins);
    for (auto [outIndex, out] : llvm::enumerate(lhs))
      outs[outIndex] = out.second;
    for (auto [outIndex, out] : llvm::enumerate(rhs))
      outs[lhs.size() + outIndex] = out.second;
  });
}

std::optional<LinearLayout>
LinearLayout::tryCompose(const LinearLayout &outer,
                         std::string *diagnostic) const {
  auto fail = [&](std::string why) -> std::optional<LinearLayout> {
    if (diagnostic)
      *diagnostic = std::move(why);
    return std::nullopt;
  };
  if (getNumOutDims() != outer.getNumInDims())
    return fail("compose: outer's in-dims differ from this layout's out-dims");
  for (auto [outDim, size] : outDims)
    if (!outer.hasInDim(outDim) || outer.getInDimSize(outDim) < size)
      return fail("compose: outer's in-dim '" + outDim.str() +
                  "' can't hold this layout's out-dim");
  int64_t numValues = getTotalInDimSize();
  if (numValues > kMaxValues)
    return fail("compose: more than 2^20 inputs to enumerate");
  std::vector<std::vector<int32_t>> values(outer.getNumOutDims(),
                                           std::vector<int32_t>(numValues));
  forEachInput(inputRadices, [&](auto ins, int64_t index) {
    for (auto [outIndex, out] : llvm::enumerate(outer.apply(apply(ins))))
      values[outIndex][index] = out.second;
  });
  return fromValues(inputRadices, llvm::to_vector(outer.outDims), values,
                    diagnostic);
}

std::optional<LinearLayout>
LinearLayout::tryInvertAndCompose(const LinearLayout &outer,
                                  std::string *diagnostic) const {
  auto fail = [&](std::string why) -> std::optional<LinearLayout> {
    if (diagnostic)
      *diagnostic = std::move(why);
    return std::nullopt;
  };
  if (getNumOutDims() != outer.getNumOutDims() ||
      llvm::any_of(getOutDimNames(),
                   [&](StringAttr d) { return !outer.hasOutDim(d); }))
    return fail("invertAndCompose: the layouts have different out-dims");
  int64_t numValues = getTotalInDimSize();
  if (numValues > kMaxValues || outer.getTotalInDimSize() > kMaxValues)
    return fail("invertAndCompose: more than 2^20 inputs to enumerate");
  // The lowest outer input reaching each output, keyed in this layout's order.
  auto key = [&](ArrayRef<std::pair<StringAttr, int32_t>> outs) {
    SmallVector<int32_t> ret;
    for (StringAttr outDim : getOutDimNames())
      ret.push_back(llvm::find_if(outs, [&](auto out) {
                      return out.first == outDim;
                    })->second);
    return ret;
  };
  std::map<SmallVector<int32_t>, SmallVector<int32_t>> preimage;
  forEachInput(outer.inputRadices, [&](auto ins, int64_t) {
    preimage.try_emplace(key(outer.apply(ins)),
                         llvm::to_vector(llvm::make_second_range(ins)));
  });
  std::vector<std::vector<int32_t>> values(outer.getNumInDims(),
                                           std::vector<int32_t>(numValues));
  std::optional<SmallVector<int32_t>> missing;
  forEachInput(inputRadices, [&](auto ins, int64_t index) {
    if (missing)
      return;
    SmallVector<int32_t> outs = key(apply(ins));
    auto it = preimage.find(outs);
    if (it == preimage.end()) {
      missing = outs;
      return;
    }
    for (auto [i, value] : llvm::enumerate(it->second))
      values[i][index] = value;
  });
  if (missing)
    return fail("invertAndCompose: output (" + join(*missing, ", ") +
                ") is outside outer's image");
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (StringAttr inDim : outer.getInDimNames())
    newOutDims.push_back({inDim, outer.getInDimSize(inDim)});
  return fromValues(inputRadices, newOutDims, values, diagnostic);
}

SmallVector<std::pair<StringAttr, int32_t>> LinearLayout::applyMixedRadix(
    ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  auto digitValue = [&](StringAttr inDim, int32_t pos) {
    uint32_t quotient =
        llvm::find_if(ins, [&](auto in) { return in.first == inDim; })->second;
    ArrayRef<int32_t> radices = getInputRadices(inDim);
    for (int32_t radix : radices.take_front(pos))
      quotient /= radix;
    return int64_t(quotient % radices[pos]);
  };
  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (auto [outIndex, outDim] : llvm::enumerate(getOutDimNames())) {
    ArrayRef<std::pair<StringAttr, int32_t>> table = getTableDigits(outDim);
    int32_t outVal = 0;
    for (const auto &[inDim, inDimBases] : bases)
      for (auto [pos, basis] : llvm::enumerate(inDimBases))
        if (basis[outIndex] != 0 &&
            !llvm::is_contained(table, std::pair(inDim, int32_t(pos))) &&
            digitValue(inDim, pos) != 0)
          outVal ^= basis[outIndex];
    if (!table.empty()) {
      int64_t index = 0, weight = 1;
      for (auto [inDim, pos] : table) {
        index += digitValue(inDim, pos) * weight;
        weight *= getInputRadix(inDim, pos);
      }
      outVal += getTableOffsets(outDim)[index];
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}

LinearLayout LinearLayout::reshapeMixedRadixIns(
    ArrayRef<std::pair<StringAttr, int32_t>> newInDims) const {
  // Flatten the digits, then cut them where the new dims' sizes are reached.
  SmallVector<std::pair<int32_t, std::vector<int32_t>>> flatDigits;
  SmallVector<std::pair<StringAttr, int32_t>> oldDigits;
  for (const auto &[inDim, inDimBases] : bases) {
    for (auto [pos, radix, basis] :
         llvm::enumerate(getInputRadices(inDim), inDimBases)) {
      flatDigits.push_back({radix, basis});
      oldDigits.push_back({inDim, int32_t(pos)});
    }
  }
  BasesT newBases;
  RadicesT newRadices;
  llvm::DenseMap<std::pair<StringAttr, int32_t>, std::pair<StringAttr, int32_t>>
      moved;
  size_t i = 0;
  for (const auto &[inDim, inDimSize] : newInDims) {
    auto &dimBases = newBases[inDim];
    auto &dimRadices = newRadices[inDim];
    int64_t size = 1;
    for (; size < inDimSize && i < flatDigits.size();
         size *= flatDigits[i++].first) {
      moved[oldDigits[i]] = {inDim, int32_t(dimRadices.size())};
      dimRadices.push_back(flatDigits[i].first);
      dimBases.push_back(flatDigits[i].second);
    }
    assert(size == inDimSize && "new in-dim sizes must end on a digit");
  }
  assert(i == flatDigits.size() && "new in-dims must take every digit");
  decltype(tableDigits) newDigits = tableDigits;
  for (auto &[outDim, digits] : newDigits)
    for (auto &digit : digits)
      digit = moved.lookup(digit);
  return LinearLayout(std::move(newBases), std::move(newRadices),
                      llvm::to_vector(outDims), std::move(newDigits),
                      tableOffsets, isSurjective());
}

std::optional<LinearLayout>
LinearLayout::resizeMixedRadixOutDim(StringAttr outDim, int32_t newSize) const {
  int32_t size = getOutDimSize(outDim);
  if (newSize == size)
    return *this;
  // Keeping f mod newSize drops the table only when it adds whole multiples
  // of newSize; size 1 drops any table.
  bool dropsTable =
      newSize == 1 ||
      (llvm::isPowerOf2_32(newSize) && newSize <= twoPartOf(size) &&
       llvm::all_of(getTableOffsets(outDim),
                    [&](int32_t offset) { return offset % newSize == 0; }));
  if (!dropsTable)
    return std::nullopt;
  BasesT newBases = bases;
  int32_t outIndex = getOutDimIndex(outDim);
  for (auto &[inDim, inDimBases] : newBases)
    for (auto &basis : inDimBases)
      if (basis[outIndex] >= newSize)
        basis[outIndex] = 0;
  auto newOutDims = llvm::to_vector(outDims);
  newOutDims[outIndex].second = newSize;
  decltype(tableDigits) newDigits = tableDigits;
  decltype(tableOffsets) newOffsets = tableOffsets;
  newDigits.erase(outDim);
  newOffsets.erase(outDim);
  return LinearLayout(std::move(newBases), inputRadices, newOutDims,
                      std::move(newDigits), std::move(newOffsets),
                      /*requireSurjective=*/false);
}

LinearLayout LinearLayout::remapDigits(const DigitMap &moved) const {
  BasesT newBases;
  RadicesT newRadices;
  for (StringAttr inDim : getInDimNames()) {
    newBases[inDim];
    newRadices[inDim];
  }
  for (const auto &[from, to] : moved) {
    auto &dimBases = newBases[to.first];
    auto &dimRadices = newRadices[to.first];
    if (dimBases.size() <= size_t(to.second)) {
      dimBases.resize(to.second + 1);
      dimRadices.resize(to.second + 1);
    }
    dimBases[to.second] = bases.find(from.first)->second[from.second];
    dimRadices[to.second] = getInputRadix(from.first, from.second);
  }
  decltype(tableDigits) digits;
  decltype(tableOffsets) offsets;
  remapTables(moved, digits, offsets);
  return LinearLayout(std::move(newBases), std::move(newRadices),
                      llvm::to_vector(outDims), std::move(digits),
                      std::move(offsets), /*requireSurjective=*/false);
}

void LinearLayout::remapTables(const DigitMap &moved,
                               decltype(tableDigits) &digits,
                               decltype(tableOffsets) &offsets) const {
  for (const auto &[outDim, oldDigits] : tableDigits) {
    ArrayRef<int32_t> oldOffsets = getTableOffsets(outDim);
    SmallVector<std::pair<StringAttr, int32_t>> newDigits;
    SmallVector<int64_t> keptWeights;
    SmallVector<int32_t> keptRadices;
    int64_t weight = 1;
    for (const auto &digit : oldDigits) {
      int32_t radix = getInputRadix(digit.first, digit.second);
      if (auto it = moved.find(digit); it != moved.end()) {
        newDigits.push_back(it->second);
        keptWeights.push_back(weight);
        keptRadices.push_back(radix);
      }
      weight *= radix;
    }
    if (newDigits.empty())
      continue;
    digits[outDim] = std::move(newDigits);
    int64_t entries = std::accumulate(keptRadices.begin(), keptRadices.end(),
                                      int64_t{1}, std::multiplies<>());
    auto &newOffsets = offsets[outDim];
    for (int64_t t = 0; t < entries; ++t) {
      int64_t quotient = t, index = 0;
      for (auto [radix, w] : llvm::zip(keptRadices, keptWeights)) {
        index += quotient % radix * w;
        quotient /= radix;
      }
      newOffsets.push_back(oldOffsets[index]);
    }
  }
}

void LinearLayout::sublayoutTables(ArrayRef<StringAttr> inDimNames,
                                   ArrayRef<StringAttr> outDimNames,
                                   decltype(tableDigits) &digits,
                                   decltype(tableOffsets) &offsets) const {
  // A dropped in-dim is held at 0, so its digits leave the table.
  DigitMap moved;
  for (StringAttr inDim : inDimNames)
    for (int32_t pos = 0, e = getInputRadices(inDim).size(); pos < e; ++pos)
      moved[{inDim, pos}] = {inDim, pos};
  remapTables(moved, digits, offsets);
  for (StringAttr outDim : getOutDimNames()) {
    if (!llvm::is_contained(outDimNames, outDim)) {
      digits.erase(outDim);
      offsets.erase(outDim);
    }
  }
}

LinearLayout
LinearLayout::removeMixedRadixZeroBases(StringAttr stripDim) const {
  // A table digit stays even when its stored basis is zero.
  DigitMap moved;
  for (const auto &[inDim, inDimBases] : bases) {
    int32_t numKept = 0;
    for (auto [pos, basis] : llvm::enumerate(inDimBases)) {
      std::pair<StringAttr, int32_t> digit(inDim, pos);
      if (inDim != stripDim ||
          llvm::any_of(basis, [](int32_t b) { return b != 0; }) ||
          llvm::any_of(tableDigits, [&](const auto &table) {
            return llvm::is_contained(table.second, digit);
          }))
        moved[digit] = {inDim, numKept++};
    }
  }
  return remapDigits(moved);
}

std::optional<LinearLayout>
LinearLayout::resizeMixedRadixInDim(StringAttr inDim, int32_t newSize) const {
  // Shrinking keeps a prefix of the digits; growing appends zero radix-2
  // digits.
  ArrayRef<int32_t> radices = getInputRadices(inDim);
  int64_t size = 1;
  int32_t numKept = 0;
  for (; numKept < int32_t(radices.size()) && size < newSize; ++numKept)
    size *= radices[numKept];
  if (size > newSize || newSize % size != 0 ||
      !llvm::isPowerOf2_64(newSize / size))
    return std::nullopt;
  if (numKept < int32_t(radices.size())) {
    DigitMap moved;
    for (const auto &[dim, dimBases] : bases)
      for (int32_t pos = 0, e = dimBases.size(); pos < e; ++pos)
        if (dim != inDim || pos < numKept)
          moved[{dim, pos}] = {dim, pos};
    return remapDigits(moved);
  }
  BasesT newBases = bases;
  RadicesT newRadices = inputRadices;
  for (int64_t s = size; s < newSize; s *= 2) {
    newBases[inDim].push_back(std::vector<int32_t>(getNumOutDims(), 0));
    newRadices[inDim].push_back(2);
  }
  return tryCreate(std::move(newBases), std::move(newRadices),
                   llvm::to_vector(outDims), tableDigits, tableOffsets,
                   /*requireSurjective=*/false);
}

llvm::MapVector<StringAttr, int32_t>
LinearLayout::getMixedRadixFreeVariableMasks() const {
  // Mask bits are digit positions; table digits are never free.
  llvm::MapVector<StringAttr, int32_t> ret =
      binaryPart().getFreeVariableMasks();
  for (const auto &[outDim, digits] : tableDigits)
    for (auto [inDim, pos] : digits)
      ret[inDim] &= ~(int32_t{1} << pos);
  return ret;
}

size_t LinearLayout::hashMixedRadix(size_t seed) const {
  for (const auto &[inDim, radices] : inputRadices)
    seed = llvm::hash_combine(
        seed, llvm::hash_combine_range(radices.begin(), radices.end()));
  for (const auto &[outDim, digits] : tableDigits) {
    ArrayRef<int32_t> offsets = getTableOffsets(outDim);
    seed = llvm::hash_combine(
        seed, outDim, llvm::hash_combine_range(digits.begin(), digits.end()),
        llvm::hash_combine_range(offsets.begin(), offsets.end()));
  }
  return seed;
}

std::string LinearLayout::tablesToString() const {
  std::string ret;
  for (const auto &[outDim, digits] : tableDigits)
    if (digits.size() != 1 ||
        !isOddPartTable(outDims.lookup(outDim), getTableOffsets(outDim)))
      ret += " - " + outDim.str() + " table over (" +
             join(digits, ", ",
                  [](auto digit) {
                    return digit.first.str() + " digit " +
                           std::to_string(digit.second);
                  }) +
             ") -> (" + join(getTableOffsets(outDim), ", ") + ")\n";
  return ret;
}

SmallVector<Value> ColumnAction::applyMixedRadix(ValueRange values) const {
  // Output digit j is input digit action[j]; dropped input digits are 0.
  SmallVector<int64_t> weights = {1};
  for (int32_t radix : inputRadices)
    weights.push_back(weights.back() * radix);
  assert(values.size() == weights.back() &&
         "Values have a different size than the ColumnAction");
  int64_t size = 1;
  for (size_t a : action)
    size *= inputRadices[a];
  SmallVector<Value> ret;
  ret.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    int64_t quotient = i, srcIdx = 0;
    for (size_t a : action) {
      srcIdx += quotient % inputRadices[a] * weights[a];
      quotient /= inputRadices[a];
    }
    ret.push_back(values[srcIdx]);
  }
  return ret;
}

LinearLayout ColumnAction::applyMixedRadix(const LinearLayout &layout) const {
  // New digit j is old digit action[j]; the dropped digits are held at 0.
  LinearLayout::DigitMap moved;
  for (const auto &[dim, dimBases] : layout.getBases())
    if (dim != inDim)
      for (int32_t pos = 0, e = dimBases.size(); pos < e; ++pos)
        moved[{dim, pos}] = {dim, pos};
  for (auto [j, a] : llvm::enumerate(action))
    moved[{inDim, int32_t(a)}] = {inDim, int32_t(j)};
  return layout.remapDigits(moved);
}

} // namespace mlir::triton
