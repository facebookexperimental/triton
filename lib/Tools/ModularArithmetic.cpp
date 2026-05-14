#include "triton/Tools/ModularArithmetic.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace mlir::triton {

int64_t intPow(int64_t a, int b) {
  assert(b >= 0 && "intPow requires non-negative exponent");
  if (b == 0) return 1;

  int64_t result = 1;
  int64_t base = a;

  while (b > 0) {
    if (b & 1) {
      result *= base;
    }
    base *= base;
    b >>= 1;
  }

  return result;
}

// gcd(0,0) returns {0, 0, 0} by convention. INT64_MIN inputs are rejected
// (also returning {0, 0, 0}) because |INT64_MIN| does not fit in int64_t,
// so std::abs(INT64_MIN) would be undefined behavior. Callers expecting to
// operate at the bit-width limit should pre-check.
ExtGCDResult extendedGCD(int64_t a, int64_t b) {
  if (a == std::numeric_limits<int64_t>::min() ||
      b == std::numeric_limits<int64_t>::min()) {
    return {0, 0, 0};
  }
  bool a_neg = a < 0;
  bool b_neg = b < 0;
  a = std::abs(a);
  b = std::abs(b);

  if (a == 0) {
    return {b, 0, b_neg ? -1 : 1};
  }
  if (b == 0) {
    return {a, a_neg ? -1 : 1, 0};
  }

  // Iterative extended GCD maintaining r_i = s_i * a + t_i * b
  int64_t r0 = a, r1 = b;
  int64_t s0 = 1, s1 = 0;
  int64_t t0 = 0, t1 = 1;

  while (r1 != 0) {
    int64_t q = r0 / r1;

    int64_t r_new = r0 - q * r1;
    r0 = r1;
    r1 = r_new;

    int64_t s_new = s0 - q * s1;
    s0 = s1;
    s1 = s_new;

    int64_t t_new = t0 - q * t1;
    t0 = t1;
    t1 = t_new;
  }

  int64_t x = a_neg ? -s0 : s0;
  int64_t y = b_neg ? -t0 : t0;

  return {r0, x, y};
}

std::optional<int64_t> modInverse(int64_t a, int64_t m) {
  if (m <= 0) return std::nullopt;

  a = ((a % m) + m) % m;
  if (a == 0) return std::nullopt;

  auto result = extendedGCD(a, m);
  if (result.gcd != 1) {
    return std::nullopt;
  }

  int64_t inv = ((result.x % m) + m) % m;
  return inv;
}

std::optional<CRTResult> solveCRT(const std::vector<int64_t> &remainders,
                                  const std::vector<int64_t> &moduli) {
  if (remainders.size() != moduli.size() || remainders.empty()) {
    return std::nullopt;
  }

  size_t n = moduli.size();

  for (auto m : moduli) {
    if (m <= 0) {
      return std::nullopt;
    }
  }

  // Pairwise coprimality is a precondition; callers ensure via factorization.
  assert([&] {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j)
        if (std::gcd(moduli[i], moduli[j]) != 1)
          return false;
    return true;
  }() && "solveCRT requires pairwise coprime moduli");

  // NB: M can overflow int64 when the product of moduli exceeds ~3e9 (squared
  // fits in int64). Callers must ensure moduli come from layout dimensions
  // which are bounded well below this in practice.
  int64_t M = 1;
  for (auto m : moduli) {
    M *= m;
  }

  // CRT: x = Σ(r_i * M_i * y_i) mod M, where M_i = M/m_i, y_i = M_i^{-1} mod m_i
  int64_t solution = 0;

  for (size_t i = 0; i < n; ++i) {
    int64_t m_i = moduli[i];
    int64_t r_i = ((remainders[i] % m_i) + m_i) % m_i;
    int64_t M_i = M / m_i;

    auto y_i_opt = modInverse(M_i, m_i);
    if (!y_i_opt.has_value()) {
      return std::nullopt;
    }
    int64_t y_i = y_i_opt.value();

    int64_t term = ((r_i % M) * (M_i % M)) % M;
    term = (term * y_i) % M;
    solution = (solution + term) % M;
  }

  solution = ((solution % M) + M) % M;
  return CRTResult{solution, M};
}

void ModMatrix::normalize() {
  // Z/N arithmetic is only defined for N > 0. A non-positive modulus would
  // make the % operations below either crash (modulus == 0) or produce
  // ill-defined sign behavior (modulus < 0). Callers must construct
  // ModMatrix with modulus > 0; assert here so misuse fails loudly.
  assert(modulus > 0 && "ModMatrix modulus must be positive");
  for (auto &val : data) {
    val = ((val % modulus) + modulus) % modulus;
  }
}

int modRREF(ModMatrix &mat) {
  int64_t mod = mat.modulus;
  int rows = mat.rows;
  int cols = mat.cols;

  // Normalize all entries first
  mat.normalize();

  int pivotRow = 0;

  for (int col = 0; col < cols && pivotRow < rows; ++col) {
    // Find pivot: row with non-zero entry in this column
    int bestPivot = -1;
    for (int r = pivotRow; r < rows; ++r) {
      if (mat.at(r, col) % mod != 0) {
        // Prefer entries coprime to mod (invertible)
        if (std::gcd(mat.at(r, col), mod) == 1) {
          bestPivot = r;
          break;
        }
        if (bestPivot == -1) {
          bestPivot = r;
        }
      }
    }

    if (bestPivot == -1) {
      // No pivot in this column, move to next column
      continue;
    }

    // Swap rows to bring pivot to pivotRow
    if (bestPivot != pivotRow) {
      for (int c = 0; c < cols; ++c) {
        std::swap(mat.at(pivotRow, c), mat.at(bestPivot, c));
      }
    }

    // Get the pivot value
    int64_t pivotVal = mat.at(pivotRow, col);

    // Try to get modular inverse
    auto invOpt = modInverse(pivotVal, mod);

    if (!invOpt.has_value()) {
      // Pivot is not invertible — can't reduce this column.
      // Swap the row back and skip without advancing pivotRow.
      if (bestPivot != pivotRow) {
        for (int c = 0; c < cols; ++c) {
          std::swap(mat.at(pivotRow, c), mat.at(bestPivot, c));
        }
      }
      continue;
    }

    int64_t pivotInv = invOpt.value();

    // Scale pivot row to make pivot = 1
    for (int c = 0; c < cols; ++c) {
      mat.at(pivotRow, c) = (mat.at(pivotRow, c) * pivotInv) % mod;
    }

    // Eliminate all other rows in this column
    for (int r = 0; r < rows; ++r) {
      if (r == pivotRow) continue;

      int64_t factor = mat.at(r, col);
      if (factor == 0) continue;

      // Subtract factor * pivotRow from row r
      for (int c = 0; c < cols; ++c) {
        int64_t val = mat.at(r, c) - factor * mat.at(pivotRow, c);
        mat.at(r, c) = ((val % mod) + mod) % mod;
      }
    }

    pivotRow++;
  }

  return pivotRow;
}

// NB: The `modulus` parameter is the modulus used for the solve, which may
// differ from A.modulus (the modulus A was originally constructed with).
// Callers (e.g. modSolveLinearCRT) re-reduce A's entries mod the new modulus.
std::vector<int64_t> modSolveLinear(const ModMatrix &A,
                                     const std::vector<int64_t> &b,
                                     int64_t modulus) {
  int n = A.rows;
  int m = A.cols;

  if (static_cast<int>(b.size()) != n) {
    return {};  // Size mismatch
  }

  // Z/N arithmetic is only defined for N > 0. The downstream % modulus
  // operations would otherwise crash (modulus == 0) or produce ill-defined
  // results (modulus < 0).
  if (modulus <= 0) {
    return {};
  }

  // Create augmented matrix [A | b]
  ModMatrix aug(n, m + 1, modulus);

  // Copy A, reducing mod the target modulus
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < m; ++c) {
      aug.at(r, c) = ((A.at(r, c) % modulus) + modulus) % modulus;
    }
  }

  // Copy b
  for (int r = 0; r < n; ++r) {
    aug.at(r, m) = ((b[r] % modulus) + modulus) % modulus;
  }

  // Perform RREF
  modRREF(aug);

  // Check for inconsistent system (0 = c where c != 0)
  for (int r = 0; r < n; ++r) {
    bool rowIsZero = true;
    for (int c = 0; c < m; ++c) {
      if (aug.at(r, c) != 0) {
        rowIsZero = false;
        break;
      }
    }
    if (rowIsZero && aug.at(r, m) != 0) {
      // Inconsistent: 0 = aug.at(r, m) where aug.at(r, m) != 0
      return {};
    }
  }

  // Extract solution
  std::vector<int64_t> x(m, 0);

  for (int r = 0; r < n; ++r) {
    // Find leading 1 in this row
    int leadCol = -1;
    for (int c = 0; c < m; ++c) {
      if (aug.at(r, c) != 0) {
        leadCol = c;
        break;
      }
    }

    if (leadCol >= 0) {
      // Should be 1 after RREF, but normalize anyway
      x[leadCol] = aug.at(r, m);
    }
  }

  return x;
}

// Solve Ax = b (mod p^e) by solving mod p then lifting each power via
// residual correction: r = (b - Ax) / p^k, delta = A^{-1} r (mod p).
// NB: intermediate products can overflow int64 for large matrices or high
// prime powers; layout dimensions keep values small enough in practice.
std::vector<int64_t> modSolveLinearHensel(const ModMatrix &A,
                                           const std::vector<int64_t> &b,
                                           int64_t prime,
                                           int exponent) {
  int n = A.rows;
  int m = A.cols;

  if (static_cast<int>(b.size()) != n) {
    return {};
  }

  if (exponent < 1) {
    return {};
  }

  // modulus = prime^exponent. We accumulate sums of products A[r,c] * x[c]
  // in int64; both factors are < modulus after reduction. To stay safe we
  // require modulus <= 2^31 so that products fit in int64 and the per-row
  // accumulation across m columns can't wrap (m bounded by tensor rank,
  // well under 1000 in practice). Triton's layout-derived moduli stay
  // below 2^20, so this leaves ~11 bits of headroom.
  int64_t modulus = intPow(prime, exponent);
  assert(modulus <= (int64_t{1} << 31) &&
         "modSolveLinearHensel: modulus too large; "
         "accumulation may overflow int64");
  (void)modulus;

  std::vector<int64_t> x_k = modSolveLinear(A, b, prime);
  if (x_k.empty()) {
    return {};
  }

  if (exponent == 1) {
    return x_k;
  }

  int64_t p_k = prime;

  for (int k = 2; k <= exponent; ++k) {
    std::vector<int64_t> residual(n);

    for (int r = 0; r < n; ++r) {
      int64_t ax_r = 0;
      for (int c = 0; c < m; ++c) {
        ax_r += A.at(r, c) * x_k[c];
      }

      int64_t diff = b[r] - ax_r;
      if (diff % p_k != 0) {
        return {};
      }
      residual[r] = diff / p_k;
    }

    std::vector<int64_t> delta = modSolveLinear(A, residual, prime);
    if (delta.empty()) {
      return {};
    }

    for (int i = 0; i < m; ++i) {
      x_k[i] += p_k * delta[i];
    }

    p_k *= prime;
  }

  int64_t p_e = intPow(prime, exponent);
  for (auto &val : x_k) {
    val = ((val % p_e) + p_e) % p_e;
  }

  return x_k;
}

std::vector<int64_t> modSolveLinearCRT(const ModMatrix &A,
                                        const std::vector<int64_t> &b,
                                        int64_t modulus) {
  int n = A.rows;
  int m = A.cols;

  if (static_cast<int>(b.size()) != n) {
    return {};
  }

  // modulus must be a positive integer for Z/N arithmetic to be defined.
  // factorize(0) returns no factors, which would leave moduli/subsolutions
  // empty and silently fail at the per-variable solveCRT step. Negative
  // moduli are similarly ill-defined.
  if (modulus <= 0) {
    return {};
  }

  if (modulus == 1) {
    return std::vector<int64_t>(m, 0);
  }

  PrimeFactorization factors = factorize(modulus);

  std::vector<std::vector<int64_t>> subsolutions;
  std::vector<int64_t> moduli;

  for (const auto &[p, e] : factors.factors) {
    int64_t mod_i = intPow(p, e);
    moduli.push_back(mod_i);

    ModMatrix A_i(n, m, mod_i);
    std::vector<int64_t> b_i(n);

    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < m; ++c) {
        A_i.at(r, c) = ((A.at(r, c) % mod_i) + mod_i) % mod_i;
      }
      b_i[r] = ((b[r] % mod_i) + mod_i) % mod_i;
    }

    std::vector<int64_t> x_i;
    if (e == 1) {
      x_i = modSolveLinear(A_i, b_i, p);
    } else {
      x_i = modSolveLinearHensel(A_i, b_i, p, e);
    }

    if (x_i.empty()) {
      return {};
    }

    subsolutions.push_back(x_i);
  }

  std::vector<int64_t> x(m);

  for (int var = 0; var < m; ++var) {
    std::vector<int64_t> remainders;
    for (const auto &sol : subsolutions) {
      remainders.push_back(sol[var]);
    }

    auto crt_result = solveCRT(remainders, moduli);
    if (!crt_result.has_value()) {
      return {};
    }

    x[var] = crt_result->solution;
  }

  return x;
}

int64_t PrimeFactorization::product() const {
  int64_t result = 1;
  for (const auto &[prime, exp] : factors)
    result *= intPow(prime, exp);
  return result;
}

PrimeFactorization factorize(int64_t n) {
  // Trial division O(sqrt(n)); for Triton's bounded moduli (typically < 2^20)
  // this completes in microseconds.
  PrimeFactorization result;

  if (n <= 1) {
    return result;
  }

  // Handle factor of 2
  int exp2 = 0;
  while (n % 2 == 0) {
    exp2++;
    n /= 2;
  }
  if (exp2 > 0) {
    result.factors.push_back({2, exp2});
  }

  // Handle odd factors
  for (int64_t i = 3; i * i <= n; i += 2) {
    int exp = 0;
    while (n % i == 0) {
      exp++;
      n /= i;
    }
    if (exp > 0) {
      result.factors.push_back({i, exp});
    }
  }

  // If n > 1, then it's a prime factor
  if (n > 1) {
    result.factors.push_back({n, 1});
  }

  return result;
}

} // namespace mlir::triton
