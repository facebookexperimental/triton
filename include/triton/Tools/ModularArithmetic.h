#ifndef TRITON_TOOLS_MODULARARITHMETIC_H
#define TRITON_TOOLS_MODULARARITHMETIC_H

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::triton {

// Result of Extended Euclidean Algorithm: (gcd, x, y) where ax + by = gcd
struct ExtGCDResult {
  int64_t gcd;
  int64_t x;
  int64_t y;
};

// Extended Euclidean Algorithm: returns gcd(a,b) and Bezout coefficients
ExtGCDResult extendedGCD(int64_t a, int64_t b);

// Modular inverse: returns x such that (a*x) % m == 1, or nullopt if gcd(a,m) != 1
std::optional<int64_t> modInverse(int64_t a, int64_t m);

// Chinese Remainder Theorem result: solution x where x ≡ r_i (mod m_i)
struct CRTResult {
  int64_t solution;
  int64_t modulus;
};

// Solve system x ≡ r_i (mod m_i). Requires pairwise coprime moduli.
std::optional<CRTResult> solveCRT(const std::vector<int64_t> &remainders,
                                  const std::vector<int64_t> &moduli);

// Integer exponentiation: a^b
int64_t intPow(int64_t a, int b);

// Matrix over Z/nZ in row-major order
struct ModMatrix {
  std::vector<int64_t> data;
  int rows;
  int cols;
  int64_t modulus; // Modulus for matrix arithmetic

  ModMatrix(int r, int c, int64_t mod) : rows(r), cols(c), modulus(mod) {
    data.resize(r * c, 0);
  }

  int64_t &at(int r, int c) { return data[r * cols + c]; }
  const int64_t &at(int r, int c) const { return data[r * cols + c]; }

  void normalize();
};

// Gaussian elimination to RREF in Z/nZ. Modifies matrix in-place, returns rank.
int modRREF(ModMatrix &mat);

// Solve Ax = b (mod modulus) using Gaussian elimination.
// The modulus parameter overrides A.modulus for the solve operation.
std::vector<int64_t> modSolveLinear(const ModMatrix &A,
                                     const std::vector<int64_t> &b,
                                     int64_t modulus);

// Solve Ax = b (mod p^e) using Hensel lifting
std::vector<int64_t> modSolveLinearHensel(const ModMatrix &A,
                                           const std::vector<int64_t> &b,
                                           int64_t prime,
                                           int exponent);

// Solve Ax = b (mod N) using CRT: factor N, solve subsystems, combine solutions.
// Precondition: modulus > 0. Returns {} on precondition violation.
std::vector<int64_t> modSolveLinearCRT(const ModMatrix &A,
                                        const std::vector<int64_t> &b,
                                        int64_t modulus);

// Prime factorization: N = p1^e1 * p2^e2 * ... * pk^ek
struct PrimeFactorization {
  std::vector<std::pair<int64_t, int>> factors;

  int64_t product() const;
};

// Compute prime factorization of n
PrimeFactorization factorize(int64_t n);

} // namespace mlir::triton

#endif // TRITON_TOOLS_MODULARARITHMETIC_H
