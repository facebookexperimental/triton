// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "Z3ConstraintEncoder.h"

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER

#include <algorithm>
#include <string>

namespace mlir::triton::gpu {
namespace {

static std::string toDecimalString(__int128 value) {
  bool negative = value < 0;
  unsigned __int128 magnitude =
      negative ? static_cast<unsigned __int128>(-(value + 1)) + 1
               : static_cast<unsigned __int128>(value);
  std::string result;
  do {
    result.push_back(static_cast<char>('0' + magnitude % 10));
    magnitude /= 10;
  } while (magnitude != 0);
  if (negative)
    result.push_back('-');
  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace

Z3ConstraintEncoder::Z3ConstraintEncoder(Z3_context context, Z3_solver solver)
    : context(context), solver(solver), intSort(Z3_mk_int_sort(context)) {}

Z3_ast Z3ConstraintEncoder::intValue(int64_t value) const {
  return Z3_mk_int64(context, value, intSort);
}

Z3_ast Z3ConstraintEncoder::wideIntValue(__int128 value) const {
  std::string storage = toDecimalString(value);
  return Z3_mk_numeral(context, storage.c_str(), intSort);
}

Z3_ast Z3ConstraintEncoder::variable(llvm::StringRef name) const {
  std::string storage = name.str();
  return Z3_mk_const(context, Z3_mk_string_symbol(context, storage.c_str()),
                     intSort);
}

Z3_ast Z3ConstraintEncoder::add(Z3_ast lhs, Z3_ast rhs) const {
  Z3_ast args[] = {lhs, rhs};
  return Z3_mk_add(context, 2, args);
}

Z3_ast Z3ConstraintEncoder::sub(Z3_ast lhs, Z3_ast rhs) const {
  Z3_ast args[] = {lhs, rhs};
  return Z3_mk_sub(context, 2, args);
}

Z3_ast Z3ConstraintEncoder::mul(Z3_ast lhs, Z3_ast rhs) const {
  Z3_ast args[] = {lhs, rhs};
  return Z3_mk_mul(context, 2, args);
}

Z3_ast Z3ConstraintEncoder::sum(llvm::ArrayRef<Z3_ast> terms) const {
  if (terms.empty())
    return intValue(0);
  if (terms.size() == 1)
    return terms.front();
  return Z3_mk_add(context, static_cast<unsigned>(terms.size()), terms.data());
}

Z3_ast Z3ConstraintEncoder::conjunction(llvm::ArrayRef<Z3_ast> terms) const {
  if (terms.empty())
    return Z3_mk_true(context);
  if (terms.size() == 1)
    return terms.front();
  return Z3_mk_and(context, static_cast<unsigned>(terms.size()), terms.data());
}

Z3_ast Z3ConstraintEncoder::disjunction(llvm::ArrayRef<Z3_ast> terms) const {
  if (terms.empty())
    return Z3_mk_false(context);
  if (terms.size() == 1)
    return terms.front();
  return Z3_mk_or(context, static_cast<unsigned>(terms.size()), terms.data());
}

Z3_ast Z3ConstraintEncoder::maximum(llvm::ArrayRef<Z3_ast> terms) const {
  if (terms.empty())
    return intValue(0);
  Z3_ast result = terms.front();
  for (Z3_ast term : terms.drop_front())
    result = maximum(result, term);
  return result;
}

Z3_ast Z3ConstraintEncoder::maximum(Z3_ast lhs, Z3_ast rhs) const {
  return Z3_mk_ite(context, Z3_mk_ge(context, lhs, rhs), lhs, rhs);
}

Z3_ast Z3ConstraintEncoder::implies(Z3_ast premise, Z3_ast conclusion) const {
  return Z3_mk_implies(context, premise, conclusion);
}

void Z3ConstraintEncoder::assertFormula(Z3_ast formula) const {
  Z3_solver_assert(context, solver, formula);
}

} // namespace mlir::triton::gpu

#endif // TRITON_ENABLE_Z3_JOINT_SOLVER
