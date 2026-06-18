#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <set>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/ModularArithmetic.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "linear_layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}

static int __builtin_ctzll(unsigned long long x) {
  unsigned long r;
  _BitScanForward64(&r, x);
  return static_cast<int>(r);
}

#endif

namespace mlir::triton {

namespace {
using BasesT = LinearLayout::BasesT;
using llvm::SmallDenseSet;
using llvm::Twine;

BasesT makeBasesMap(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases) {
  BasesT ret;
  for (const auto &[inDim, inDimBases] : bases) {
    ret[inDim] = inDimBases;
  }
  return ret;
}

// Dump the matrix to stderr in a human-readable format for debugging.
void dumpMatrix(uint64_t *m, int numRows, int numCols) {
  assert(numCols <= 64);
  for (int r = 0; r < numRows; r++) {
    llvm::errs() << "0b";
    for (int c = 0; c < numCols; c++) {
      llvm::errs() << ((m[r] & (1 << c)) != 0 ? "1" : "0");
    }
    llvm::errs() << "\n";
  }
}

// Compute the rank of the matrix formed by taking the bases for the given
// outDim as columns.  In other words, finds the number of linearly-independent
// bases for this output dimension.
int getMatrixRank(std::unique_ptr<uint64_t[]> m, int numRows, int numCols) {
  // stride is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  assert(numCols <= 64);
  f2reduce::inplace_rref_strided(m.get(), numRows, numCols, /*stride=*/1);

  // The rank of the reduced matrix is simply the number of nonzero rows.
  int rank = 0;
  for (int i = 0; i < numRows; i++) {
    if (m[i] != 0)
      rank++;
  }
  return rank;
}

template <typename T, typename U>
void assertDimsEqualIgnoringOrder(T &&a, U &&b) {
  SmallDenseSet<StringAttr> as(a.begin(), a.end());
  SmallDenseSet<StringAttr> bs(b.begin(), b.end());
  if (as != bs) {
    llvm::report_fatal_error("Dimensions must match, ignoring order, but they "
                             "don't.  Got dims: [" +
                             Twine(triton::join(a, ", ")) + "] and [" +
                             triton::join(b, ", ") + "]");
  }
}

template <typename T, typename U>
void assertDimsSubsetIgnoringOrder(T &&small, U &&big) {
  SmallDenseSet<StringAttr> smallSet(small.begin(), small.end());
  SmallDenseSet<StringAttr> bigSet(big.begin(), big.end());
  if (!llvm::set_is_subset(smallSet, bigSet)) {
    llvm::report_fatal_error("Dimensions must be a subset, ignoring order, but "
                             "they aren't.  Got dims: [" +
                             Twine(triton::join(small, ", ")) + "] and [" +
                             triton::join(big, ", ") + "]");
  }
}

// Build integer basis matrix from LinearLayout in row-major order.
// Each row is a basis vector; each column is an output dimension.
// Used by lstsqModular for modular arithmetic solving (as opposed to
// getMatrix() which builds the GF(2) bit-matrix for f2reduce).
std::vector<int64_t> getIntegerBasisMatrix(const LinearLayout &L) {
  int numRows = L.getTotalInDimSizeLog2();
  int numCols = L.getNumOutDims();
  std::vector<int64_t> mat(numRows * numCols, 0);

  int row = 0;
  for (auto inDim : L.getInDimNames()) {
    for (int i = 0; i < L.getInDimSizeLog2(inDim); i++) {
      auto basis = L.getBasis(inDim, i);
      for (int col = 0; col < numCols; col++) {
        mat[row * numCols + col] = basis[col];
      }
      row++;
    }
  }

  return mat;
}

} // anonymous namespace

// Forward declarations for surjectivity helpers used in checkInvariants
bool checkPow2Surjectivity(const std::vector<int32_t> &bases, int64_t modulus);
bool checkOddPrimePowerSurjectivity(const std::vector<int32_t> &bases,
                                    int64_t modulus);

/*static*/ std::optional<LinearLayout>
LinearLayout::tryCreate(BasesT bases,
                        ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                        bool requireSurjective) {
  LinearLayout ll(std::move(bases), std::move(outDims), NoCheckInvariants{});
  std::optional<std::string> error = ll.checkInvariants(requireSurjective);
  if (error) {
    return std::nullopt;
  }
  return ll;
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           NoCheckInvariants)
    : bases(std::move(bases)) {
  for (auto [outDim, size] : outDims) {
    this->outDims[outDim] = size;
  }
}

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
    : bases(std::move(bases)) {
  // Infer out-dim sizes.
  for (StringAttr outDim : outDimNames) {
    outDims[outDim] = 1;
  }
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        int32_t &size = outDims[outDimNames[i]];
        size = std::max<int32_t>(size, llvm::NextPowerOf2(basis[i]));
      }
    }
  }

  std::optional<std::string> error =
      checkInvariants(/*requireSurjective=*/true);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           bool requireSurjective)
    : LinearLayout(std::move(bases), std::move(outDims), NoCheckInvariants{}) {
  std::optional<std::string> error = checkInvariants(requireSurjective);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

std::optional<std::string>
LinearLayout::checkInvariants(bool requireSurjective) {
  LDBG("checkInvariants: " << toString());

  // Cache isModular result FIRST, before any validation that uses it.
  // This is computed once during construction since outDims is immutable.
  this->cachedIsModular = false;
  for (const auto &[outDim, size] : outDims) {
    if (!llvm::isPowerOf2_32(size)) {
      this->cachedIsModular = true;
      break;
    }
  }

  // Check that basis values are non-negative.
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t b) { return b < 0; })) {
        return "Invalid bases passed to LinearLayout.  Expected all basis "
               "values to be non-negative, but found a negative value for "
               "in dimension '" +
               inDim.str() + "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Check that the bases all have length equal to outDimNames.size().
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (basis.size() != outDims.size()) {
        return "Invalid bases passed to LinearLayout.  Expect all bases to "
               "have the same size, equal to outDimNames.size() (" +
               std::to_string(outDims.size()) +
               ").  But this failed for in dimension '" + inDim.str() +
               "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Per-dim validation: pow2 dims enforce strict basis < size (XOR semantics),
  // NPOT dims allow bases >= size (modular reduction via ADD+UREM).
  SmallVector<StringAttr> outDimNames = llvm::to_vector(getOutDimNames());
  for (const auto &[outDim, size] : outDims) {
    if (size <= 0) {
      return "Invalid out-dim size " + std::to_string(size) + " for out-dim '" +
             outDim.str() + "'.  Out-dim sizes must be positive.\n";
    }
  }

  // For pow2 dims, bases must be < size (XOR-based).
  // For NPOT dims, bases may exceed size (modular reduction handles it).
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        if (llvm::isPowerOf2_32(outDims[outDimNames[i]]) &&
            basis[i] >= outDims[outDimNames[i]]) {
          return "Invalid basis " + std::to_string(basis[i]) + " for in-dim '" +
                 inDim.str() + "' and out-dim '" + outDimNames[i].str() +
                 "'.  Basis must be less than the out-dim size.\n";
        }
      }
    }
  }

  // Determine whether the this layout is surjective, i.e. that every `out`
  // coordinate can be reached by some `in` coordinate.
  //
  // It's prohibitively slow to calculate this naively, but thankfully, this
  // is equivalent to checking that the number of linearly-independent bases
  // is equal to sum(getOutDimSizeBits).  This can be computed by finding
  // the rank of the matrix whose columns are those bases.  We can compute
  // the rank of our matrix using Gaussian elimination, which runs in O(n^3)
  // for an n x n matrix.  Our matrix size is sum(inDimSizeLog2) x
  // sum(outDimSizeBits), so this should be plenty fast.
  if (!cachedIsModular) {
    this->rank =
        getMatrixRank(getMatrix(*this), /*numRows=*/getTotalOutDimSizeBits(),
                      /*numCols=*/getTotalInDimSizeLog2());
  } else {
    // GF(2) rank is not meaningful for modular layouts.
    this->rank = -1;
  }

  if (requireSurjective && !isSurjective()) {
    return "Layout is expected to be surjective, i.e. every `out` coordinate "
           "can be reached by some `in` coordinate, but was not:" +
           toString();
  }

  return std::nullopt;
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames)
    : LinearLayout(makeBasesMap(bases), outDimNames) {}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<std::pair<StringAttr, int32_t>> outDims, bool requireSurjective)
    : LinearLayout(makeBasesMap(bases), outDims, requireSurjective) {}

/*static*/ LinearLayout LinearLayout::strided1D(int32_t size, int32_t stride,
                                                StringAttr inDimName,
                                                StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  // strided1D/identity1D are pow2-only; NPOT callers must use modularStrided1D.
  assert(llvm::isPowerOf2_32(size) &&
         "strided1D/identity1D require a power-of-2 size");
  std::vector<std::vector<int32_t>> bases;
  for (int32_t i = 1; i < size; i *= 2) {
    bases.emplace_back(std::vector<int32_t>{i * stride});
  }
  bool requireSurjective = (stride == 1);
  return LinearLayout({{inDimName, std::move(bases)}},
                      {{outDimName, stride * size}}, requireSurjective);
}

/*static*/ LinearLayout LinearLayout::zeros1D(int32_t size,
                                              StringAttr inDimName,
                                              StringAttr outDimName,
                                              int32_t outDimSize) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size) && "zeros1D requires a power-of-2 size");
  std::vector<std::vector<int32_t>> zeros;
  for (int i = 1; i < size; i *= 2) {
    zeros.emplace_back(std::vector<int32_t>{0});
  }
  return LinearLayout({{inDimName, zeros}}, {{outDimName, outDimSize}},
                      /*requireSurjective=*/outDimSize == 1);
}

/*static*/ LinearLayout LinearLayout::modularStrided1D(int32_t size,
                                                       int32_t stride,
                                                       StringAttr inDimName,
                                                       StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(size > 0 && "size must be positive");
  assert(stride > 0 && "stride must be positive");

  // Generate ceil(log2(size)) basis vectors
  int numBases = llvm::Log2_64_Ceil(size);

  std::vector<std::vector<int32_t>> bases;
  for (int i = 0; i < numBases; i++) {
    // Basis value: (stride * 2^i) % size
    // Use 1LL to avoid UB if a future caller passes a layout with > 2^31
    // elements per input bit.
    int32_t basisValue = static_cast<int32_t>(
        (static_cast<int64_t>(stride) * (1LL << i)) % size);
    bases.emplace_back(std::vector<int32_t>{basisValue});
  }

  // requireSurjective = false: modular layouts check surjectivity via
  // isModularSurjective rather than the GF(2) RREF-based check.
  return LinearLayout({{inDimName, std::move(bases)}}, {{outDimName, size}},
                      /*requireSurjective=*/false);
}

int32_t LinearLayout::getOutDimIndex(StringAttr outDim) const {
  int i = 0;
  for (auto [name, _] : outDims) {
    if (name == outDim) {
      return i;
    }
    i++;
  }
  llvm::report_fatal_error("outDim " + Twine(outDim) + " is not in layout" +
                           toString());
}

int32_t LinearLayout::getInDimSizeLog2(StringAttr inDim) const {
  auto it = bases.find(inDim);
  assert(it != bases.end() && "inDim not found in layout");
  return it->second.size();
}

int32_t LinearLayout::getTotalInDimSizeLog2() const {
  return std::accumulate(getInDimNames().begin(), getInDimNames().end(), 0,
                         [&](int32_t acc, StringAttr inDim) {
                           return acc + getInDimSizeLog2(inDim);
                         });
}

int32_t LinearLayout::getOutDimSizeLog2(StringAttr outDim) const {
  auto it = outDims.find(outDim);
  assert(it != outDims.end() && "outDim not found in layout");
  assert(llvm::isPowerOf2_32(it->second) &&
         "getOutDimSizeLog2 called on non-pow2 out-dim - use "
         "getOutDimSizeBits() instead");
  return llvm::Log2_32(it->second);
}

int32_t LinearLayout::getTotalOutDimSizeLog2() const {
  assert(!isModular() && "getTotalOutDimSizeLog2 called on modular layout - "
                         "use getTotalOutDimSizeBits() instead");
  return std::accumulate(getOutDimNames().begin(), getOutDimNames().end(), 0,
                         [&](int32_t acc, StringAttr outDim) {
                           return acc + getOutDimSizeLog2(outDim);
                         });
}

// Check surjectivity for a single dimension modulo a power of 2.
//
// CORRECTNESS: apply() composes modular (NPOT) out-dims with ADD+mod (Z/N
// arithmetic), NOT XOR (GF(2)). The previous GF(2)/XOR-rank check therefore
// reported surjectivity for the wrong group structure: e.g. bases {1,3} mod 4
// have full GF(2) rank (XOR reaches all of {0..3}) but their *subset sums*
// mod 4 only reach {0,1,3} — value 2 is unreachable. This made non-surjective
// layouts (e.g. bases {1,3,4,8} over size 12, which reaches only 9 of 12
// values under ADD+mod) wrongly pass isSurjective(). Use the same additive
// dynamic-programming reachability check as checkOddPrimePowerSurjectivity so
// the surjectivity test matches apply()'s ADD+mod semantics.
// Complexity: O(numBases * modulus) time, O(modulus) space.
bool checkPow2Surjectivity(const std::vector<int32_t> &bases, int64_t modulus) {
  assert(llvm::isPowerOf2_64(modulus) && "modulus must be power of 2");

  // Filter out zero bases — they don't contribute to reachability.
  std::vector<int32_t> nonZeroBases;
  for (int32_t b : bases) {
    if (b % modulus != 0) {
      nonZeroBases.push_back(b);
    }
  }

  int numBases = nonZeroBases.size();
  if (numBases == 0) {
    return modulus == 1;
  }

  // DP reachability over subset sums mod `modulus`, matching apply()'s ADD+mod
  // composition of basis contributions. Uses two vectors with swap to avoid
  // copying per iteration.
  std::vector<bool> reachable(modulus, false);
  std::vector<bool> next(modulus, false);
  reachable[0] = true;
  int reachableCount = 1;
  for (int i = 0; i < numBases && reachableCount < modulus; i++) {
    int32_t basis = ((nonZeroBases[i] % modulus) + modulus) % modulus;
    next = reachable;
    for (int64_t v = 0; v < modulus; v++) {
      if (reachable[v] && !next[(v + basis) % modulus]) {
        next[(v + basis) % modulus] = true;
        reachableCount++;
      }
    }
    std::swap(reachable, next);
  }

  // Check if all values mod modulus are reachable.
  for (int64_t i = 0; i < modulus; i++) {
    if (!reachable[i]) {
      return false;
    }
  }
  return true;
}

// Check surjectivity modulo an odd prime power via DP reachability.
// Complexity: O(numBases * modulus) time, O(modulus) space.
bool checkOddPrimePowerSurjectivity(const std::vector<int32_t> &bases,
                                    int64_t modulus) {
  // Filter out zero bases — they don't contribute to reachability.
  // Critical for multi-dimensional layouts where bases from unrelated
  // input dimensions are all zero for the current output dimension.
  std::vector<int32_t> nonZeroBases;
  for (int32_t b : bases) {
    if (b % modulus != 0) {
      nonZeroBases.push_back(b);
    }
  }

  int numBases = nonZeroBases.size();
  if (numBases == 0) {
    return modulus == 1;
  }

  // DP reachability: O(numBases * modulus) — correct for all basis patterns.
  // Uses two vectors with swap to avoid copying per iteration.
  std::vector<bool> reachable(modulus, false);
  std::vector<bool> next(modulus, false);
  reachable[0] = true;
  int reachableCount = 1;
  for (int i = 0; i < numBases && reachableCount < modulus; i++) {
    int32_t basis = nonZeroBases[i] % modulus;
    next = reachable;
    for (int64_t v = 0; v < modulus; v++) {
      if (reachable[v] && !next[(v + basis) % modulus]) {
        next[(v + basis) % modulus] = true;
        reachableCount++;
      }
    }
    std::swap(reachable, next);
  }

  // Check if all values mod modulus are reachable
  for (int64_t i = 0; i < modulus; i++) {
    if (!reachable[i]) {
      return false;
    }
  }
  return true;
}

bool LinearLayout::isModularSurjective() const {
  assert(isModular() && "isModularSurjective() requires a modular layout; "
                        "use isSurjective() for pow2 layouts");

  // For modular layouts, we need to check if all values [0, size) can be
  // reached using modular addition of bases. For multi-dimensional layouts,
  // check each output dimension independently (product space is surjective iff
  // each dimension is).

  // Iterate over all output dimensions and check each independently
  for (const auto &[outDim, size] : outDims) {
    int32_t outDimIdx = getOutDimIndex(outDim);

    // Collect all basis values for this output dimension
    std::vector<int32_t> allBases;
    for (const auto &[inDim, inDimBases] : bases) {
      for (const auto &basis : inDimBases) {
        allBases.push_back(basis[outDimIdx]);
      }
    }

    int numBases = allBases.size();
    if (numBases == 0) {
      // Empty layout, only covers value 0
      if (size != 1) {
        return false;
      }
      continue; // Check next dimension
    }

    // Use CRT factorization: check surjectivity per prime power factor.
    // For N = 2^k * p1^e1 * ... * pm^em, we check each factor separately.
    // Each factor uses DP reachability, so the total cost is
    // O(numBases * sum_i(pi^ei)) rather than the naive O(2^numBases).
    auto factorization = factorize(size);

    for (const auto &[prime, exponent] : factorization.factors) {
      int64_t modulus = intPow(prime, exponent);

      // Reduce all bases modulo this factor
      std::vector<int32_t> basesModFactor;
      for (int32_t base : allBases) {
        basesModFactor.push_back(((base % modulus) + modulus) % modulus);
      }

      bool surjective;
      if (prime == 2) {
        // Power of 2: use GF(2) rank check (O(numBases^3))
        surjective = checkPow2Surjectivity(basesModFactor, modulus);
      } else {
        // Odd prime power: DP reachability, O(numBases * modulus)
        surjective = checkOddPrimePowerSurjectivity(basesModFactor, modulus);
      }

      if (!surjective) {
        return false;
      }
    }
  }

  // All dimensions are surjective
  return true;
}

int32_t LinearLayout::getNumConsecutiveInOut() const {
  if (bases.empty() || getNumOutDims() == 0)
    return 1;

  // Count how many of the initial bases for the first in-dim are
  // (2^i, 0, ..., 0).
  const auto &firstInDimBases = bases.begin()->second;
  int consec = 0;
  for (; consec < firstInDimBases.size(); consec++) {
    const auto &basis = firstInDimBases[consec];
    if (basis[0] != (1 << consec) ||
        !std::all_of(basis.begin() + 1, basis.end(),
                     [](int32_t x) { return x == 0; })) {
      break;
    }
  }

  // `or` together all other bases' first out-dim.
  int32_t otherBits = 0;
  for (const auto &[inDim, inDimBases] : bases) {
    for (int i = 0; i < inDimBases.size(); i++) {
      if (inDim != bases.begin()->first || i >= consec) {
        otherBits |= inDimBases[i][0];
      }
    }
  }
  int32_t trailingZeros = otherBits != 0 ? __builtin_ctz(otherBits) : 31;

  int32_t result = 1 << std::min(consec, trailingZeros);
  // For NPOT (modular) output dims, clamp to avoid mod-N wrap-around.
  // Max safe V = largest pow2 dividing N (2-adic valuation).
  StringAttr firstOutDim = outDims.begin()->first;
  if (isOutDimModular(firstOutDim)) {
    int32_t n = getOutDimSize(firstOutDim);
    result = std::min(result, n & (-n));
  }
  return result;
}

LinearLayout LinearLayout::transposeIns(ArrayRef<StringAttr> newInDims) const {
  assertDimsEqualIgnoringOrder(newInDims, getInDimNames());

  BasesT newBases;
  for (const auto &inDim : newInDims) {
    newBases[inDim] = bases.find(inDim)->second;
  }
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
                      isSurjective());
}

LinearLayout
LinearLayout::transposeOuts(ArrayRef<StringAttr> newOutDims) const {
  assertDimsEqualIgnoringOrder(newOutDims, getOutDimNames());

  std::vector<int32_t> permutation;
  for (const auto &outDim : newOutDims) {
    permutation.push_back(getOutDimIndex(outDim));
  }

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      std::vector<int32_t> newBasis;
      for (int32_t i : permutation) {
        newBasis.push_back(basis[i]);
      }
      newInDimBases.push_back(std::move(newBasis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> newOutDimSizes;
  for (auto outDim : newOutDims) {
    newOutDimSizes.push_back({outDim, getOutDimSize(outDim)});
  }
  return LinearLayout(std::move(newBases), newOutDimSizes, isSurjective());
}

LinearLayout LinearLayout::reshapeIns(
    ArrayRef<std::pair<StringAttr, int32_t>> newInDims) const {
  assert(llvm::all_of(newInDims, [&](auto &inDim) {
    return llvm::isPowerOf2_32(inDim.second);
  }));
  assert(getTotalInDimSize() == std::accumulate(newInDims.begin(),
                                                newInDims.end(), 1,
                                                [&](int32_t acc, auto &inDim) {
                                                  return acc * inDim.second;
                                                }));

  // First flatten into a single in-dimension.  Then split it up according
  // to `newInDims`.
  SmallVector<std::vector<int32_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      flatBases.push_back(basis);
    }
  }

  BasesT newBases;
  int i = 0;
  for (const auto &[inDim, inDimSize] : newInDims) {
    auto &newInDimBases = newBases[inDim];
    for (int j = 1; j < inDimSize; j *= 2) {
      newInDimBases.push_back(flatBases[i++]);
    }
  }
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
                      isSurjective());
}

LinearLayout LinearLayout::reshapeOuts(
    ArrayRef<std::pair<StringAttr, int32_t>> newOutDims) const {
  assert(getTotalOutDimSizeProduct() ==
         std::accumulate(
             newOutDims.begin(), newOutDims.end(), int64_t{1},
             [&](int64_t acc, auto &outDim) { return acc * outDim.second; }));

  // Mixed-radix encoding: use multipliers instead of bit-shifts.
  // For pow2 dims, multiply is equivalent to shift. For NPOT, it's correct.
  SmallVector<int64_t> multipliers;
  multipliers.push_back(1);
  for (StringAttr outDim : getOutDimNames()) {
    multipliers.push_back(multipliers.back() * getOutDimSize(outDim));
  }

  // Flatten into a single out-dimension.  Then split it up according to
  // `newOutDims`.
  llvm::MapVector<StringAttr, std::vector<int64_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &flatInBases = flatBases[inDim];
    for (const auto &basis : inDimBases) {
      int64_t b = 0;
      for (int i = 0; i < basis.size(); i++) {
        // Widen to int64 before multiplying: the flattened mixed-radix index
        // can exceed int32 range for large layouts even though each basis[i]
        // is int32. The widening is intentional overflow prevention.
        b += static_cast<int64_t>(basis[i]) * multipliers[i];
      }
      flatInBases.push_back(b);
    }
  }

  BasesT newBases;
  for (const auto &[inDim, flatInBases] : flatBases) {
    std::vector<std::vector<int32_t>> &newInDimBases = newBases[inDim];
    for (int64_t b : flatInBases) {
      std::vector<int32_t> multiDimBasis;
      for (int32_t newSize : llvm::make_second_range(newOutDims)) {
        multiDimBasis.push_back(b % newSize);
        b /= newSize;
      }
      newInDimBases.push_back(std::move(multiDimBasis));
    }
  }

  return LinearLayout(std::move(newBases), newOutDims, isSurjective());
}

LinearLayout LinearLayout::resizeInDim(StringAttr inDim,
                                       int32_t newSize) const {
  assert(llvm::isPowerOf2_32(newSize));
  assert(newSize <= getInDimSize(inDim));
  auto newBases = bases;
  newBases[inDim].resize(llvm::Log2_32(newSize));
  return LinearLayout(std::move(newBases), getOutDims(),
                      /*requireSurjective=*/false);
}

LinearLayout LinearLayout::resizeOutDim(StringAttr outDim,
                                        int32_t newSize) const {
  assert(newSize <= getOutDimSize(outDim));
  auto newBases = bases;
  // Zero-out the basis vectors that are greater than or equal to the new size
  for (auto &[inDim, inDimBases] : newBases) {
    for (auto &basis : inDimBases) {
      auto &b = basis[getOutDimIndex(outDim)];
      if (b >= newSize) {
        b = 0;
      }
    }
  }
  auto outDims = getOutDims();
  for (auto &[dim, size] : outDims) {
    if (dim == outDim) {
      size = newSize;
    }
  }
  return LinearLayout(std::move(newBases), outDims,
                      /*requireSurjective=*/false);
}

LinearLayout LinearLayout::concatIns(const LinearLayout &other) const {
  assert(llvm::to_vector(getOutDimNames()) ==
             llvm::to_vector(other.getOutDimNames()) &&
         "layouts must have the same output dimensions");
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) == other.getOutDimSize(outDim) &&
           "layouts must have the same output dimension sizes");
  }

  LinearLayout::BasesT resultBases = getBases();
  for (auto &bases : other.getBases())
    resultBases.insert(bases);
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto &[outDim, outDimSize] : outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  return LinearLayout(std::move(resultBases), newOutDims,
                      /*requireSurjective=*/false);
}

LinearLayout LinearLayout::concatOuts(const LinearLayout &other) const {
  assert(llvm::to_vector(getInDimNames()) ==
             llvm::to_vector(other.getInDimNames()) &&
         "layouts must have the same input dimensions");
  for (StringAttr inDim : getInDimNames()) {
    assert(getInDimSize(inDim) == other.getInDimSize(inDim) &&
           "layouts must have the same input dimension sizes");
  }

  LinearLayout::BasesT result;
  for (auto [lhsBases, rhsBases] : llvm::zip(getBases(), other.getBases())) {
    auto &resultBases = result[lhsBases.first];
    assert(lhsBases.first == rhsBases.first);
    for (auto [lhsBasis, rhsBasis] :
         llvm::zip(lhsBases.second, rhsBases.second)) {
      std::vector<int32_t> resultBasis;
      llvm::append_range(resultBasis, lhsBasis);
      llvm::append_range(resultBasis, rhsBasis);
      resultBases.push_back(std::move(resultBasis));
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto &[outDim, outDimSize] : outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  for (auto &[outDim, outDimSize] : other.outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  return LinearLayout(std::move(result), newOutDims,
                      /*requireSurjective=*/false);
}

std::optional<LinearLayout> divideLeft(const LinearLayout &A,
                                       const LinearLayout &B) {
  // Compute a C such that A = B * C if it exists.
  // Note that such a C exists iff (every pair of input/output dim of) A is of
  // the form
  // [[B, 0],
  //  [0, C]]
  // as a matrix, whenever those dimensions are present in B.
  for (StringAttr dim : B.getInDimNames()) {
    if (!llvm::is_contained(A.getInDimNames(), dim))
      return std::nullopt;
  }
  for (StringAttr dim : B.getOutDimNames()) {
    if (!llvm::is_contained(A.getOutDimNames(), dim))
      return std::nullopt;
  }
  // Compute candidate C’s sizes for output dimensions.
  llvm::MapVector<StringAttr, int32_t> cOutDimSizes;
  for (StringAttr outDim : A.getOutDimNames()) {
    int sizeA = A.getOutDimSize(outDim);
    int sizeB = B.hasOutDim(outDim) ? B.getOutDimSize(outDim) : 1;
    if (sizeA % sizeB != 0)
      return std::nullopt;
    cOutDimSizes[outDim] = sizeA / sizeB;
  }

  LinearLayout::BasesT cBases;
  for (StringAttr inDim : A.getInDimNames()) {
    int inA = A.getInDimSizeLog2(inDim);
    int inB = B.hasInDim(inDim) ? B.getInDimSizeLog2(inDim) : 0;
    int inC = inA - inB;
    if (inC < 0)
      return std::nullopt;

    std::vector<std::vector<int32_t>> basesForDim;
    // Check that A’s first inB entries agree with B.
    for (int i = 0; i < inB; ++i) {
      for (StringAttr outDim : A.getOutDimNames()) {
        int expected = B.hasOutDim(outDim) ? B.getBasis(inDim, i, outDim) : 0;
        int actual = A.getBasis(inDim, i, outDim);
        if (actual != expected)
          return std::nullopt;
      }
    }

    // Extract the candidate C bases from the remaining entries in A.
    // For A = B * C, outer bases are multiplied by B’s out-dim size.
    for (int i = inB; i < inA; ++i) {
      std::vector<int32_t> candidateBasis;
      for (StringAttr outDim : llvm::make_first_range(cOutDimSizes)) {
        int sizeB = B.hasOutDim(outDim) ? B.getOutDimSize(outDim) : 1;
        int v = A.getBasis(inDim, i, outDim);

        // v must be divisible by B’s out-dim size.
        if (v % sizeB != 0)
          return std::nullopt;
        candidateBasis.push_back(v / sizeB);
      }
      basesForDim.push_back(std::move(candidateBasis));
    }
    cBases[inDim] = basesForDim;
  }

  SmallVector<std::pair<StringAttr, int32_t>> COutDims;
  for (auto [outDim, outC] : cOutDimSizes) {
    COutDims.push_back({outDim, outC});
  }
  // If the layout A and B are surjective, then C should also be surjective.
  LinearLayout C(std::move(cBases), COutDims,
                 /*requireSurjective=*/A.isSurjective() && B.isSurjective());
  assert(B * C == A);
  return C;
}

std::optional<LinearLayout> divideRight(const LinearLayout &A,
                                        const LinearLayout &B) {
  // Compute a C such that A = C * B if it exists.
  // Note that such a C exists iff (every pair of input/output dim of) A is of
  // the form
  // [[C, 0],
  //  [0, B]]
  // as a matrix, whenever those dimensions are present in B.

  // Check that B's in-dimensions and out-dimensions are contained in A.
  for (StringAttr dim : B.getInDimNames()) {
    if (!llvm::is_contained(A.getInDimNames(), dim))
      return std::nullopt;
  }
  for (StringAttr dim : B.getOutDimNames()) {
    if (!llvm::is_contained(A.getOutDimNames(), dim))
      return std::nullopt;
  }

  // Compute candidate C's sizes for output dimensions.
  llvm::MapVector<StringAttr, int32_t> cOutDimSizes;
  for (StringAttr outDim : A.getOutDimNames()) {
    int sizeA = A.getOutDimSize(outDim);
    int sizeB = B.hasOutDim(outDim) ? B.getOutDimSize(outDim) : 1;
    if (sizeA % sizeB != 0)
      return std::nullopt;
    cOutDimSizes[outDim] = sizeA / sizeB;
  }

  // For candidate C, its in-dim sizes come from subtracting B's in-dim sizes
  // from A's.
  LinearLayout::BasesT cBases;
  for (StringAttr inDim : A.getInDimNames()) {
    int inA = A.getInDimSizeLog2(inDim);
    int inB = B.hasInDim(inDim) ? B.getInDimSizeLog2(inDim) : 0;
    int inC = inA - inB;
    if (inC < 0)
      return std::nullopt;

    std::vector<std::vector<int32_t>> basesForDim;
    // The first inC basis vectors come directly from C.
    for (int i = 0; i < inC; ++i) {
      std::vector<int32_t> candidate;
      for (StringAttr outDim : llvm::make_first_range(cOutDimSizes)) {
        candidate.push_back(A.getBasis(inDim, i, outDim));
      }
      basesForDim.push_back(std::move(candidate));
    }

    // The remaining inB basis vectors in A should correspond to B, scaled
    // by C's out-dim size (since A = C * B, B is the outer layout).
    for (int i = inC; i < inA; ++i) {
      int j = i - inC; // Index into B's basis vectors for this inDim.
      for (StringAttr outDim : B.getOutDimNames()) {
        int sizeC = cOutDimSizes[outDim];
        int v = A.getBasis(inDim, i, outDim);
        // v must be divisible by C's out-dim size.
        if (v % sizeC != 0)
          return std::nullopt;
        int recovered = v / sizeC;
        int expected = B.getBasis(inDim, j, outDim);
        if (recovered != expected)
          return std::nullopt;
      }
    }
    cBases[inDim] = basesForDim;
  }

  SmallVector<std::pair<StringAttr, int32_t>> COutDims;
  for (auto [outDim, size] : cOutDimSizes)
    COutDims.push_back({outDim, size});
  // If A and B are surjective, then C should also be surjective.
  LinearLayout C(std::move(cBases), COutDims,
                 /*requireSurjective=*/A.isSurjective() && B.isSurjective());
  assert(C * B == A);
  return C;
}

LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that dims common to outer and inner have the same relative order.
  auto inDims = supremum(llvm::to_vector(inner.getInDimNames()),
                         llvm::to_vector(outer.getInDimNames()));
  auto outDims = supremum(llvm::to_vector(inner.getOutDimNames()),
                          llvm::to_vector(outer.getOutDimNames()));

  // Track input dim sizes as log2 (always pow2) and output dim sizes as actual
  // values (may be NPOT). For shared output dims, the combined size is the
  // product of inner and outer sizes (mixed-radix encoding).
  llvm::MapVector<StringAttr, int32_t> inDimSizesLog2;
  llvm::MapVector<StringAttr, int32_t> outDimActualSizes;
  for (const auto &dim : inDims)
    inDimSizesLog2.insert({dim, 0});
  for (const auto &dim : outDims)
    outDimActualSizes.insert({dim, 1});
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimActualSizes[outDim] *= layout.getOutDimSize(outDim);
    }
  }

  BasesT allBases;
  for (auto [inDimName, inDimSizeLog2] : inDimSizesLog2) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSizeLog2, std::vector<int32_t>(outDimActualSizes.size(), 0));

    for (auto [outDimIdx, outDimNameAndSize] :
         llvm::enumerate(outDimActualSizes)) {
      auto [outDimName, outDimSize] = outDimNameAndSize;
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset =
            inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;
        // Use multiply instead of shift: correct for both pow2 and NPOT.
        // For pow2, multiply by 2^k is equivalent to << k.
        int multiplier =
            inner.hasOutDim(outDimName) ? inner.getOutDimSize(outDimName) : 1;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          inDimBases[offset + i][outDimIdx] =
              outer.getBasis(inDimName, i, outDimName) * multiplier;
        }
      }
    }
  }

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (auto [outDim, actualSize] : outDimActualSizes) {
    outDimSizes.push_back({outDim, actualSize});
  }
  return LinearLayout(std::move(allBases), outDimSizes,
                      inner.isSurjective() && outer.isSurjective());
}

bool LinearLayout::isTrivialOver(ArrayRef<StringAttr> dimNames) const {
  for (StringAttr dim : dimNames) {
    if (!llvm::is_contained(getInDimNames(), dim) &&
        !llvm::is_contained(getOutDimNames(), dim)) {
      return false;
    }
  }

  auto getRemainingDimNames = [&](auto allDimNames) {
    SmallVector<StringAttr> remainingDimNames;
    for (StringAttr dim : allDimNames) {
      if (!llvm::is_contained(dimNames, dim)) {
        remainingDimNames.push_back(dim);
      }
    }
    return remainingDimNames;
  };
  SmallVector<StringAttr> remainingInDimNames =
      getRemainingDimNames(getInDimNames());
  SmallVector<StringAttr> remainingOutDimNames =
      getRemainingDimNames(getOutDimNames());

  // Think of this as a block-matrix multiplying a vector:
  // [[A, B],  *  [v_1,
  //  [C, D]]      v_2]
  // where v_2 is the dimNames and v_1 is the remainingInDimNames
  // We can quotient out dimNames iff they don't affect the remainingInDimNames
  // in the result. In other words, we want to check that B is zero, and C is
  // zero, and D is the identity
  return squareSublayoutIsIdentity(*this, dimNames) &&
         sublayoutIsZero(remainingInDimNames, dimNames) &&
         sublayoutIsZero(dimNames, remainingOutDimNames);
}

std::optional<LinearLayout>
LinearLayout::quotient(ArrayRef<StringAttr> dimNames) const {
  if (!isTrivialOver(dimNames)) {
    return std::nullopt;
  }

  // This should probably be even less general, where we ask inDimNames ==
  // outDimNames
  auto getRemainingDimNames = [&](auto allDimNames) {
    SmallVector<StringAttr> remainingDimNames;
    for (StringAttr dim : allDimNames) {
      if (!llvm::is_contained(dimNames, dim)) {
        remainingDimNames.push_back(dim);
      }
    }
    return remainingDimNames;
  };

  SmallVector<StringAttr> inDimNames = getRemainingDimNames(getInDimNames());
  SmallVector<StringAttr> outDimNames = getRemainingDimNames(getOutDimNames());

  return sublayout(inDimNames, outDimNames);
}

LinearLayout LinearLayout::sublayout(ArrayRef<StringAttr> inDimNames,
                                     ArrayRef<StringAttr> outDimNames) const {
  assertDimsSubsetIgnoringOrder(inDimNames, getInDimNames());
  assertDimsSubsetIgnoringOrder(outDimNames, getOutDimNames());
  SmallDenseSet<StringAttr> inDimSet(inDimNames.begin(), inDimNames.end());
  SmallDenseSet<StringAttr> outDimSet(outDimNames.begin(), outDimNames.end());

  SmallVector<int> outDimIndicesToKeep;
  for (auto [i, outDim] : llvm::enumerate(getOutDimNames())) {
    if (outDimSet.contains(outDim)) {
      outDimIndicesToKeep.push_back(i);
    }
  }
  BasesT newBases;
  for (auto [inDim, inDimBases] : bases) {
    if (!inDimSet.contains(inDim)) {
      continue;
    }
    auto &newInDimBases = newBases[inDim];
    for (auto &basis : inDimBases) {
      auto &newBasis = newInDimBases.emplace_back();
      for (int i : outDimIndicesToKeep) {
        newBasis.push_back(basis[i]);
      }
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto [outDim, outDimSize] : outDims) {
    if (outDimSet.contains(outDim)) {
      newOutDims.push_back({outDim, outDimSize});
    }
  }
  return LinearLayout(std::move(newBases), std::move(newOutDims),
                      /*requireSurjective=*/false);
}

bool LinearLayout::sublayoutIsZero(ArrayRef<StringAttr> inDimNames,
                                   ArrayRef<StringAttr> outDimNames) const {
  LinearLayout ss = sublayout(inDimNames, outDimNames);
  for (auto [inDim, inDimBases] : ss.bases) {
    for (auto basis : inDimBases) {
      if (!llvm::all_of(basis, [](int32_t b) { return b == 0; })) {
        return false;
      }
    }
  }
  return true;
}

SmallVector<std::pair<StringAttr, int32_t>>
LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  assertDimsEqualIgnoringOrder(llvm::make_first_range(ins), getInDimNames());

  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (StringAttr outDim : getOutDimNames()) {
    int32_t outVal = 0;
    // Per-dim dispatch: NPOT dims use ADD+mod (Z/N arithmetic), pow2 dims use
    // XOR (GF(2)). With the split-dim layout, pow2 dims (dim0, contig_intra)
    // are solved via GF(2) RREF while the NPOT dim (contig_phase) is solved via
    // modSolveLinearCRT. S=0 between groups means no coupling, so per-dim
    // dispatch is correct.
    bool dimIsModular = isOutDimModular(outDim);
    for (auto &[inDim, val] : ins) {
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        if (val & (1 << i)) {
          if (dimIsModular) {
            outVal =
                (outVal + getBasis(inDim, i, outDim)) % getOutDimSize(outDim);
          } else {
            outVal ^= getBasis(inDim, i, outDim);
          }
        }
      }
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}

LinearLayout LinearLayout::compose(const LinearLayout &outer) const {
  assertDimsEqualIgnoringOrder(getOutDimNames(), outer.getInDimNames());
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) <= outer.getInDimSize(outDim));
  }

  bool compositionIsSurjective =
      isSurjective() && outer.isSurjective() &&
      llvm::all_of(getOutDimNames(), [&](StringAttr outDim) {
        return getOutDimSize(outDim) == outer.getInDimSize(outDim);
      });

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      SmallVector<std::pair<StringAttr, int32_t>> bases;
      for (auto [outDim, b] : llvm::zip(getOutDimNames(), basis)) {
        bases.push_back({outDim, b});
      }
      auto newBases = outer.apply(bases);
      auto newBasesRange = llvm::make_second_range(newBases);
      newInDimBases.push_back(
          std::vector<int32_t>(newBasesRange.begin(), newBasesRange.end()));
    }
  }

  return LinearLayout(std::move(newBases), llvm::to_vector(outer.outDims),
                      compositionIsSurjective);
}

std::unique_ptr<uint64_t[]>
LinearLayout::concatMatrices(const LinearLayout &A, const LinearLayout &B) {
  // conv
  assert(A.getTotalOutDimSizeBits() >= B.getTotalOutDimSizeBits() &&
         "A must have at least as many output bits as B");
  int numColsA = A.getTotalInDimSizeLog2();

  // rref expects the lower bits to be the lower indices of the matrix
  auto concat = getMatrix(A);
  auto BMat = getMatrix(B);
  int rowA = 0;
  int rowB = 0;
  for (auto outDim : A.getOutDimNames()) {
    // Use getOutDimSizeBits() to handle NPOT dimensions correctly
    for (int r = 0; r < A.getOutDimSizeBits(outDim); r++) {
      if (B.hasOutDim(outDim) && r < B.getOutDimSizeBits(outDim)) {
        concat[rowA] |= BMat[rowB] << numColsA;
        rowB++;
      }
      rowA++;
    }
  }
  return concat;
}

// Modular lstsq implementation delegating to modSolveLinearCRT
LinearLayout LinearLayout::lstsqModular(const LinearLayout &A,
                                        const LinearLayout &B) {
  assertDimsEqualIgnoringOrder(A.getOutDimNames(), B.getOutDimNames());

  int numRowsA = A.getTotalInDimSizeLog2();
  int numRowsB = B.getTotalInDimSizeLog2();
  int numOutDims = A.getNumOutDims();

  // Working modulus = LCM of all output dimension sizes
  int64_t workingModulus = 1;
  for (auto [outDim, size] : A.getOutDims()) {
    int64_t g = std::gcd(workingModulus, static_cast<int64_t>(size));
    workingModulus = (workingModulus / g) * size;
  }

  auto matA_data = getIntegerBasisMatrix(A);
  auto matB_data = getIntegerBasisMatrix(B);

  // Build ModMatrix for A (numOutDims rows x numRowsA cols)
  ModMatrix matA_mod(numOutDims, numRowsA, workingModulus);
  for (int r = 0; r < numRowsA; r++)
    for (int c = 0; c < numOutDims; c++)
      matA_mod.at(c, r) = matA_data[r * numOutDims + c];

  // Solve per column of B
  std::vector<std::vector<int64_t>> resultCols(numRowsB);
  for (int bCol = 0; bCol < numRowsB; bCol++) {
    std::vector<int64_t> rhs(numOutDims);
    for (int outIdx = 0; outIdx < numOutDims; outIdx++)
      rhs[outIdx] = matB_data[bCol * numOutDims + outIdx];

    resultCols[bCol] = modSolveLinearCRT(matA_mod, rhs, workingModulus);
    assert(!resultCols[bCol].empty() &&
           "Precondition broken in lstsqModular: modSolveLinearCRT failed. "
           "Im(B) not contained in Im(A)?");
  }

  // Build result layout from solution
  assert(!A.getInDimNames().empty() &&
         "A must have at least one input dimension");
  StringAttr inDim1D = *B.getInDimNames().begin();
  StringAttr outDim1D = *A.getInDimNames().begin();

  LinearLayout::BasesT retBases;
  auto &bs = retBases[inDim1D];

  for (int bCol = 0; bCol < numRowsB; bCol++) {
    std::vector<int32_t> basis(1);
    basis[0] = 0;

    for (int aRow = 0; aRow < numRowsA; aRow++) {
      if (resultCols[bCol][aRow] != 0) {
        basis[0] = (basis[0] + resultCols[bCol][aRow] * (int64_t{1} << aRow)) %
                   static_cast<int32_t>(A.getTotalInDimSize());
      }
    }

    bs.push_back(basis);
  }

  LinearLayout retFlattened(std::move(retBases),
                            {{outDim1D, A.getTotalInDimSize()}},
                            /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : B.getInDimNames()) {
    retInDims.push_back({dim, B.getInDimSize(dim)});
  }
  for (StringAttr dim : A.getInDimNames()) {
    retOutDims.push_back({dim, A.getInDimSize(dim)});
  }
  return retFlattened.reshapeIns(retInDims).reshapeOuts(retOutDims);
}

LinearLayout LinearLayout::lstsq(const LinearLayout &A, const LinearLayout &B) {
  // Dispatch to modular solver if needed.
  // Only use the modular solver when A (the layout being inverted) is modular.
  // When only B is modular (e.g., NPOT output dim from msgToPackedOffset) but A
  // is pow2 (e.g., smemLayout with pow2 alloc shape), the GF(2) solver is both
  // correct and necessary: the modular solver's working modulus (LCM of output
  // dim sizes) can be much smaller than A's input space, causing it to confuse
  // distinct offsets that map to different coordinates. The GF(2) solver works
  // correctly because it operates on individual bits and finds the exact input
  // bit combination that produces B's output.
  if (A.isModular()) {
    // Mixed pow2/NPOT output dims: solve each group with its correct algebra.
    // The split-dim construction (dim_intra=pow2, dim_phase=NPOT) guarantees
    // S=0 cross-coupling, so the groups are algebraically independent.
    SmallVector<StringAttr> pow2OutDims, npotOutDims;
    for (auto [outDim, size] : A.getOutDims()) {
      if (llvm::isPowerOf2_32(size))
        pow2OutDims.push_back(outDim);
      else
        npotOutDims.push_back(outDim);
    }

    if (!pow2OutDims.empty() && !npotOutDims.empty()) {
      auto inDimNames = llvm::to_vector(A.getInDimNames());
      auto bInDimNames = llvm::to_vector(B.getInDimNames());

      auto C_pow2 = lstsq(A.sublayout(inDimNames, pow2OutDims),
                          B.sublayout(bInDimNames, pow2OutDims));
      auto C_npot = lstsqModular(A.sublayout(inDimNames, npotOutDims),
                                 B.sublayout(bInDimNames, npotOutDims));

      // Combine: OR the per-basis coefficients (disjoint by S=0 decoupling).
      auto C_pow2_flat = C_pow2.flattenIns().flattenOuts();
      auto C_npot_flat = C_npot.flattenIns().flattenOuts();
      StringAttr flatIn = *C_pow2_flat.getInDimNames().begin();
      StringAttr flatOut = *C_pow2_flat.getOutDimNames().begin();

      int numColsB = B.getTotalInDimSizeLog2();
      LinearLayout::BasesT combinedBases;
      auto &cbs = combinedBases[flatIn];
      for (int c = 0; c < numColsB; c++) {
        int32_t vp = C_pow2_flat.getBasis(flatIn, c, flatOut);
        int32_t vn = C_npot_flat.getBasis(flatIn, c, flatOut);
        // The pow2 (XOR) and NPOT (ADD) solves are supposed to be disjoint
        // (S=0 decoupling). If any output bit is claimed by both solves, the
        // factored OR-combine would conflate them, so fall back to a full
        // modular solve.
        bool crossCoupled = (vp & vn) != 0;
        if (crossCoupled) {
          LDBG("lstsq: mixed solve fallback (cross-coupled dims)");
          return lstsqModular(A, B);
        }
        cbs.push_back({vp | vn});
      }

      LinearLayout ret(std::move(combinedBases),
                       {{flatOut, A.getTotalInDimSize()}},
                       /*requireSurjective=*/false);
      SmallVector<std::pair<StringAttr, int32_t>> retInDims, retOutDims;
      for (StringAttr dim : bInDimNames)
        retInDims.push_back({dim, B.getInDimSize(dim)});
      for (StringAttr dim : inDimNames)
        retOutDims.push_back({dim, A.getInDimSize(dim)});
      return ret.reshapeIns(retInDims).reshapeOuts(retOutDims);
    }

    return lstsqModular(A, B);
  }

  // Original GF(2) implementation
  // Solve the least square system AX = B
  // and return the least square solution X by computing RREF and setting
  // the free variables to zero.
  // A and B may not be surjective, but we assume that Im(B) \subset Im(A)
  // Sketch of the algorithm:
  // https://github.com/triton-lang/triton/pull/5309#discussion_r1869084111
  int numRows = A.getTotalOutDimSizeBits();
  assert(numRows >= B.getTotalOutDimSizeBits() &&
         "A.lstsq(B) called with incompatible output shapes");
  int numColsA = A.getTotalInDimSizeLog2();
  int numColsB = B.getTotalInDimSizeLog2();
  int numCols = numColsA + numColsB;
  std::unique_ptr<uint64_t[]> combinedMat = concatMatrices(A, B);
  f2reduce::inplace_rref_strided(combinedMat.get(), numRows, numCols,
                                 /*stride=*/1);

  // Compute the pivot columns
  // Since A and B have the same image, each row will either have a pivot
  // or will be all zeros
  SmallVector<int32_t> pivotRowOfCol(numColsA, -1);
  for (int r = 0; r < numRows; r++) {
    auto row = combinedMat[r];
    if (row == 0) {
      continue;
    }
    int c = __builtin_ctzll(row);
    assert(c < numColsA && "Precondition broken. Im(B) not contained in Im(A)");
    assert(pivotRowOfCol[c] == -1 &&
           "duplicate pivot => matrix not in RREF or A not injective");
    pivotRowOfCol[c] = r;
  }

  // Extract A^{-1}B and complete the matrix using zeros
  std::unique_ptr<uint64_t[]> retMat(new uint64_t[numColsA]());
  for (int c = 0; c < numColsA; ++c) {
    int row = pivotRowOfCol[c];
    retMat[c] = (row == -1) ? 0 : (combinedMat[row] >> numColsA);
  }

  // We need names for the in/out dim of the flattened layout we're going to
  // read off from `m`.  These could be anything, doesn't matter.
  assert(!A.getInDimNames().empty() &&
         "attempt to solve lstsq for empty layout");
  StringAttr inDim1D = *A.getInDimNames().begin();
  StringAttr outDim1D = *A.getOutDimNames().begin();

  // Read off the new bases.  These are for a flattened 1D -> 1D
  LinearLayout::BasesT retBases;
  auto &bs = retBases[inDim1D];
  for (int c = 0; c < numColsB; c++) {
    int32_t basis = 0;
    for (int r = 0; r < numColsA; r++) {
      basis |= (retMat[r] >> c & 1) << r;
    }
    bs.push_back({basis});
  }

  LinearLayout retFlattened(std::move(retBases),
                            {{outDim1D, A.getTotalInDimSize()}},
                            /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : B.getInDimNames()) {
    retInDims.push_back({dim, B.getInDimSize(dim)});
  }
  for (StringAttr dim : A.getInDimNames()) {
    retOutDims.push_back({dim, A.getInDimSize(dim)});
  }
  return retFlattened.reshapeIns(retInDims).reshapeOuts(retOutDims);
}

LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const {
  // TODO(Lezcano) Make friend and perhaps rename to `convertFrom` or `lstsq`
  // For this, we need to implement our LLVM lowerings by inverting the "outer"
  // layout, and then iterating over the elements from the "this" layout and
  // fetching the corresponding element from the "outer" layout. This exercises
  // the broadcasting that we incentivise via choosing the minimum norm solution
  // in lstsq.

  // The order of dims does not matter. We choose to transpose outer
  auto outDims = llvm::to_vector(getOutDimNames());
  assertDimsEqualIgnoringOrder(outDims, outer.getOutDimNames());
  const auto &B = *this;
  const auto A = outer.transposeOuts(outDims);
  for (auto dim : outDims) {
    assert(A.getOutDimSize(dim) >= B.getOutDimSize(dim) &&
           ("A.invertAndCompose(B) called with incompatible output shapes in " +
            dim.str() + ": " + std::to_string(A.getOutDimSize(dim)) +
            " >= " + std::to_string(B.getOutDimSize(dim)))
               .c_str());
  }

  // Broadcasting heuristic
  // Imagine we have two layouts with `warps = [[0, 0],  [0, 0]]`
  // (broadcasting) on both layouts. We could map any warp to any warp in the
  // conversion. Now, we want to map them as the identity map, to mark that
  // nothing needs to be done there (`lstsq` would map all the warps to the
  // zero warp, minimum norm solution). The heuristic here is as follows:
  // - If a dimension is the same for both layouts, we want to map it as the
  // identity
  //   Equivalently, we don't add it to the conversion
  // - Otherwise, we just call lstsq (i.e. map all the equivalent elements
  //   to the same input element) to take advantage of broadcasting in shared
  //   memory and avoid saving repeated elements in shared memory

  // FIXME: We should check that the other dimensions don't touch the image of
  // this dimension.
  SmallVector<StringAttr> identityDims;
  for (auto dim : A.getInDimNames()) {
    if (B.hasInDim(dim) &&
        A.sublayout(dim, outDims) == B.sublayout(dim, outDims)) {
      identityDims.push_back(dim);
    }
  }
  SmallVector<StringAttr> ANonIdentityInDims;
  SmallVector<StringAttr> BNonIdentityInDims;
  for (auto dim : A.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      ANonIdentityInDims.push_back(dim);
    }
  }
  for (auto dim : B.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      BNonIdentityInDims.push_back(dim);
    }
  }

  auto AReduced = A.sublayout(ANonIdentityInDims, outDims);
  auto BReduced = B.sublayout(BNonIdentityInDims, outDims);

  // If one is empty, the other must be empty as well
  assert((ANonIdentityInDims.empty()) == (BNonIdentityInDims.empty()));
  bool isEmpty = ANonIdentityInDims.empty();

  auto ret = isEmpty ? LinearLayout::empty() : lstsq(AReduced, BReduced);

  // --- NPOT kernel equivalence fix ---
  //
  // For NPOT modular layouts with pure modular solving (no shear family),
  // the lstsq result maps into A's input space of size 2^k, but A's output
  // has only N < 2^k distinct values. Multiple inputs mapping to the same
  // output (kernel elements) may get different result values. Reducing the
  // output dim from 2^k to N folds kernel elements via mod-N arithmetic.
  //
  // The replacement layout declares its output dim as the NPOT size N, which
  // makes it a modular layout: subsequent apply() calls use ADD+mod (Z/N)
  // instead of XOR. This is why we only require each reduced basis to be a
  // pow2 or zero (condition 1) and rely on the reachability check at the
  // end (A(x) == A(x mod N) for all x) to validate the fold. We deliberately
  // do NOT enforce the stricter pairwise-disjoint / XOR-sum-in-[0,N)
  // conditions used by tryReduceBasesModN in TritonGPUToLLVM/Utility.cpp:
  // those guard XOR-based SMEM emission, where the layout is later combined
  // via XOR; here the resulting layout is itself modular, so XOR-overlap
  // semantics are not in play. See the kernel-equivalence tests
  // (PseudoinvertModular_KernelEquivalence,
  // InvertAndCompose_KernelEquivalence_Store) for the cases (e.g. N=48)
  // where the looser check is necessary.
  if (!isEmpty && AReduced.isModular() && ret.getNumOutDims() == 1) {
    int64_t effectiveOutSize = AReduced.getTotalOutDimSizeProduct();
    int64_t totalInSize = AReduced.getTotalInDimSize();
    if (effectiveOutSize < totalInSize) {
      StringAttr outDim = *ret.getOutDimNames().begin();
      int32_t N = static_cast<int32_t>(effectiveOutSize);

      LinearLayout::BasesT npotBases;
      bool basesNonOverlapping = true;
      for (auto &[inDim, inDimBases] : ret.getBases()) {
        auto &newBases = npotBases[inDim];
        for (auto &basis : inDimBases) {
          int32_t val = basis[0] % N;
          newBases.push_back({val});
          if (val != 0 && !llvm::isPowerOf2_32(val))
            basesNonOverlapping = false;
        }
      }

      if (basesNonOverlapping) {
        auto AFlat = AReduced.flattenIns();
        StringAttr aIn = *AFlat.getInDimNames().begin();
        bool valid = true;
        // O(2^k) where k = ceil(log2(N)) for NPOT dim N. K=48 gives 16
        // iterations, K=96 gives 32. Max ~100 for any planned shape.
        // Triple-guarded by iteration cap, convergence check, and result
        // validation.
        for (int x = N; x < AFlat.getTotalInDimSize() && valid; x++) {
          if (AFlat.apply({{aIn, x}}) != AFlat.apply({{aIn, x % N}}))
            valid = false;
        }
        if (valid) {
          ret = LinearLayout(std::move(npotBases), {{outDim, N}},
                             /*requireSurjective=*/false);
        }
      }
    }
  }

  // TODO(Lezcano): We should return the reduced layout instead of re-adding the
  // identity maps. With this, we'll be able to kill `minimalCvtLayout`

  // Add the identity maps for the dimensions that are the same for both layouts
  for (auto dim : identityDims) {
    auto size = A.getInDimSize(dim);
    if (llvm::isPowerOf2_32(size)) {
      ret *= LinearLayout::identity1D(size, dim, dim);
    } else {
      ret *= LinearLayout::modularIdentity1D(size, dim, dim);
    }
  }

  // Reorder the dimensions in the result to match the order expected by the
  // current and outer layouts.
  return ret.transposeIns(llvm::to_vector(B.getInDimNames()))
      .transposeOuts(llvm::to_vector(A.getInDimNames()));
}

LinearLayout LinearLayout::invert() const {
  assert(isInvertible() &&
         "A linear layout must be surjective and square to be invertible");
  return pseudoinvert();
}

LinearLayout LinearLayout::pseudoinvert() const {
  LinearLayout identity = LinearLayout::empty();
  for (auto outDim : getOutDimNames()) {
    auto size = getOutDimSize(outDim);
    if (llvm::isPowerOf2_32(size)) {
      identity *= LinearLayout::identity1D(size, outDim, outDim);
    } else {
      identity *= LinearLayout::modularIdentity1D(size, outDim, outDim);
    }
  }
  return identity.invertAndCompose(*this);
}

LinearLayout LinearLayout::unsqueezeIn(StringAttr dim) const {
  assert(getInDimSize(dim) == 1);
  SmallVector<std::pair<StringAttr, int32_t>> newInDims;
  for (auto inDim : getInDimNames()) {
    if (inDim != dim) {
      newInDims.push_back({inDim, getInDimSize(inDim)});
    }
  }
  return reshapeIns(newInDims);
}

LinearLayout LinearLayout::unsqueezeOut(StringAttr dim) const {
  assert(getOutDimSize(dim) == 1);
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto [outDim, outDimSize] : getOutDims()) {
    if (outDim != dim) {
      newOutDims.push_back({outDim, outDimSize});
    }
  }
  return LinearLayout(bases, newOutDims, isSurjective());
}

llvm::MapVector<StringAttr, int32_t>
LinearLayout::getFreeVariableMasks() const {
  // For modular layouts, GF(2) RREF is algebraically wrong (Z/N null space !=
  // GF(2) null space). Use conservative zero-basis check instead.
  if (isModular()) {
    llvm::MapVector<StringAttr, int32_t> ret;
    for (StringAttr inDim : getInDimNames()) {
      int32_t mask = 0;
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        bool allZero = true;
        for (StringAttr outDim : getOutDimNames()) {
          if (getBasis(inDim, i, outDim) != 0) {
            allZero = false;
            break;
          }
        }
        if (allZero)
          mask |= (1 << i);
      }
      ret[inDim] = mask;
    }
    return ret;
  }

  std::unique_ptr<uint64_t[]> mat = getMatrix(*this);
  int numRows = getTotalOutDimSizeBits();
  int numCols = getTotalInDimSizeLog2();

  // stride is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  assert(numCols <= 64);
  f2reduce::inplace_rref_strided(mat.get(), numRows, numCols, /*stride=*/1);

  // For each row in the RREF matrix, identify the column with the first "1".
  // These columns correspond to the basic (i.e. non-free) variables.
  std::set<int32_t> basicVars;
  for (int r = 0; r < numRows; r++) {
    if (mat[r] == 0) {
      continue;
    }
    basicVars.insert(__builtin_ctzll(mat[r]));
  }

  llvm::MapVector<StringAttr, int32_t> ret;
  int c = 0;
  for (StringAttr dim : getInDimNames()) {
    int32_t mask = 0;
    for (int i = 0; i < getInDimSizeLog2(dim); i++, c++) {
      if (basicVars.count(c) == 0) {
        mask |= (1 << i);
      }
    }
    ret[dim] = mask;
  }
  return ret;
}

LinearLayout LinearLayout::removeZeroBasesAlongDim(StringAttr stripDim) const {
  LinearLayout::BasesT result;
  for (auto &[inDim, inDimBases] : getBases()) {
    auto &newInDimBases = result[inDim];
    if (inDim != stripDim) {
      newInDimBases = inDimBases;
      continue;
    }
    for (auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t val) { return val != 0; })) {
        newInDimBases.push_back(basis);
      }
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> newOutDimSizes;
  for (auto outDim : getOutDimNames()) {
    newOutDimSizes.push_back({outDim, getOutDimSize(outDim)});
  }
  auto newLayout = LinearLayout(std::move(result), ArrayRef(newOutDimSizes),
                                this->isSurjective());
  return newLayout;
}

size_t hash_value(const LinearLayout &layout) {
  size_t seed = 0;

  // Hash the bases
  for (const auto &base : layout.getBases()) {
    // Hash the input dimension name
    seed = llvm::hash_combine(seed, base.first);

    // Hash the vectors in bases
    for (const auto &vec : base.second) {
      for (int32_t val : vec) {
        seed = llvm::hash_combine(seed, val);
      }
    }
  }

  // Hash the output dimensions and their sizes
  for (const auto &outDim : layout.getOutDimNames()) {
    seed = llvm::hash_combine(seed, outDim, layout.getOutDimSize(outDim));
  }
  // Don't hash the surjective flag as it's a cached property
  return seed;
}

bool operator==(const LinearLayout &lhs, const LinearLayout &rhs) {
  if (!lhs.equalIgnoringOutDimSizes(rhs))
    return false;

  for (const auto &[lhsOutDimAndSize, rhsOutDimAndSize] :
       llvm::zip(lhs.outDims, rhs.outDims)) {
    if (lhsOutDimAndSize.second != rhsOutDimAndSize.second)
      return false;
  }
  return true;
}

bool LinearLayout::equalIgnoringOutDimSizes(const LinearLayout &other) const {
  // llvm::MapVector doesn't have an operator== :(.
  if (llvm::to_vector(this->getOutDimNames()) !=
      llvm::to_vector(other.getOutDimNames()))
    return false;
  if (this->bases.size() != other.bases.size())
    return false;
  for (auto it1 = this->bases.begin(), it2 = other.bases.begin();
       it1 != this->bases.end(); ++it1, ++it2) {
    if (*it1 != *it2)
      return false;
  }
  return true;
}

std::string LinearLayout::toString() const {
  // Start with a newline because we print out a bulleted list; it doesn't
  // make sense for the first line of this list to be on the same line as
  // any previous text.
  std::string ret = "\n";
  std::string outDimsStr =
      "[" +
      join(outDims, ", ",
           [](auto dimAndSize) {
             auto [outDim, size] = dimAndSize;
             return outDim.str() + " (size " + std::to_string(size) + ")";
           }) +
      "]";

  if (bases.empty()) {
    if (outDims.empty()) {
      return "\n(empty layout)";
    } else {
      return "\n(empty layout with out-dims " + outDimsStr + ")";
    }
  }

  // TODO: Add spaces for alignment.
  for (const auto &[inDim, inDimBases] : bases) {
    if (inDimBases.empty()) {
      ret += " - " + inDim.str() + " is a size 1 dimension\n";
      continue;
    }

    ret += " - " +
           join(llvm::seq(inDimBases.size()), "\n   ",
                [&, &inDim = inDim, &inDimBases = inDimBases](int i) {
                  return inDim.str() + "=" + std::to_string(1 << i) + " -> (" +
                         join(inDimBases[i], ", ") + ")";
                }) +
           "\n";
  }
  ret += "where out dims are: " + outDimsStr;
  return ret;
}

LinearLayout ColumnAction::apply(const LinearLayout &layout) const {
  assert(layout.hasInDim(inDim));
  assert(layout.getInDimSizeLog2(inDim) == inSizeLog2 &&
         "Layout has a different size than the ColumnAction");
  if (m_isIdentity) {
    return layout;
  }

  auto bases = layout.getBases();
  const auto &basesInDim = bases[inDim];
  std::vector<std::vector<int32_t>> newBases;
  newBases.reserve(action.size());
  for (size_t a : action) {
    newBases.push_back(basesInDim[a]);
  }
  bases[inDim] = std::move(newBases);

  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (auto outDim : layout.getOutDimNames()) {
    outDims.emplace_back(outDim, layout.getOutDimSize(outDim));
  }
  return LinearLayout(std::move(bases), std::move(outDims),
                      /*requireSurjective=*/false);
}

SmallVector<Value> ColumnAction::apply(ValueRange values) const {
  assert(values.size() == (1 << inSizeLog2) &&
         "Values have a different size than the ColumnAction");
  assert(inDim.str() == "register" && "Values are in registers, so we can only "
                                      "apply ColumnAction to registers");
  if (m_isIdentity) {
    return values;
  }
  auto permLL = apply(LinearLayout::identity1D(values.size(), inDim, inDim));
  SmallVector<Value> ret;
  ret.reserve(permLL.getInDimSize(inDim));
  for (int i = 0; i < permLL.getInDimSize(inDim); i++) {
    int32_t srcIdx = permLL.apply({{inDim, i}}).begin()->second;
    ret.push_back(values[srcIdx]);
  }
  return ret;
}

ColumnAction ColumnAction::leftCompose(const ColumnAction &other) const {
  assert(inDim == other.inDim);
  assert(inSizeLog2 == other.inSizeLog2);
  assert(action.size() == other.action.size());
  auto newAction = SmallVector<size_t>(action.size());
  for (size_t i = 0; i < action.size(); i++) {
    newAction[i] = action[other.action[i]];
  }
  return ColumnAction(newAction, inDim, inSizeLog2);
}

ColumnAction ColumnAction::inverse() const {
  auto invPerm = SmallVector<size_t>(action.size());
  for (size_t i = 0; i < action.size(); i++) {
    invPerm[action[i]] = i;
  }
  return ColumnAction(invPerm, inDim, inSizeLog2);
}

std::string ColumnAction::toString() const {
  std::string ret = "ColumnAction([";
  ret += join(action, ", ");
  ret += "], " + inDim.str() + ", " + std::to_string(inSizeLog2) + ")";
  return ret;
}

// Build a matrix of size sum(outDimSizeBits) x sum(inDimSizeLog2) representing
// the bases of the given layout.  This can then be used by f2reduce.
//
// This function is called from the constructor of LinearLayout, so be careful
// not to use any functions that create LLs in here.
std::unique_ptr<uint64_t[]>
LinearLayout::getMatrix(const LinearLayout &layout) {
  // For NPOT dimensions, use ceil(log2) to avoid truncating high-order bits
  int numRows = layout.getTotalOutDimSizeBits();
  int numCols = layout.getTotalInDimSizeLog2();

  // Don't handle giant LLs.  This makes some things easier; for example, each
  // row can be a single uint64_t.
  assert(numCols <= 64 && "LinearLayout too large");
  assert(numRows <= 64 && "LinearLayout too large");

  // Suppose we have a layout specified by the following values.
  //
  //   L(0,1) = (0b01, 0b1)
  //   L(0,2) = (0b10, 0b0)
  //   L(1,0) = (0b10, 0b0)
  //   L(2,0) = (0b11, 0b0)
  //
  // We will create one column per entry above.  The max bit width of the
  // codomain is (2,1), so our matrix will have 2+1=3 rows.  The final matrix
  // will be
  //
  //  | L(0,1)[0] L(0,2)[0] L(1,0)[0] L(2,0)[0] |   | 0b1001 |
  //  |    ↓         ↓         ↓         ↓      |   | 0b0111 |
  //  | L(0,1)[1] L(0,2)[1] L(1,0)[1] L(2,0)[1] | = | 0b1000 |
  //  |    ↓         ↓         ↓         ↓      |
  //
  // Note `new uint64_t[n]()` is zero-initialized, but `new uint64_t[n]` is not.
  std::unique_ptr<uint64_t[]> m(new uint64_t[numRows]());
  int r = 0;
  for (StringAttr outDim : layout.getOutDimNames()) {
    int c = 0;
    for (StringAttr inDim : layout.getInDimNames()) {
      for (int i = 0; i < layout.getInDimSizeLog2(inDim); i++) {
        uint64_t basis = layout.getBasis(inDim, i, outDim);
        // Extract all bits needed for this dimension
        for (int j = 0; j < layout.getOutDimSizeBits(outDim); j++) {
          m[r + j] |= ((basis >> j) & 1) << c;
        }
        c++;
      }
    }
    r += layout.getOutDimSizeBits(outDim);
  }

  return m;
}

} // namespace mlir::triton
