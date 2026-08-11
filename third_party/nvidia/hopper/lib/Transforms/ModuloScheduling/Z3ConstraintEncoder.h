// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_CONSTRAINT_ENCODER_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_CONSTRAINT_ENCODER_H

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

#if __has_include("third-party/z3/4_15_2/src/api/z3.h")
#include "third-party/z3/4_15_2/src/api/z3.h"
#elif __has_include(<z3.h>)
#include <z3.h>
#elif __has_include(<api/z3.h>)
#include <api/z3.h>
#else
#error "Z3 headers are unavailable"
#endif

namespace mlir::triton::gpu {

/// Typed facade for constructing integer Z3 constraints used by the native
/// modulo-scheduling models. Context and solver ownership stays with the
/// caller; this class only creates expressions and records assertions.
class Z3ConstraintEncoder {
public:
  Z3ConstraintEncoder(Z3_context context, Z3_solver solver);

  Z3_ast intValue(int64_t value) const;
  Z3_ast wideIntValue(__int128 value) const;
  Z3_ast variable(llvm::StringRef name) const;

  Z3_ast add(Z3_ast lhs, Z3_ast rhs) const;
  Z3_ast sub(Z3_ast lhs, Z3_ast rhs) const;
  Z3_ast mul(Z3_ast lhs, Z3_ast rhs) const;
  Z3_ast sum(llvm::ArrayRef<Z3_ast> terms) const;
  Z3_ast conjunction(llvm::ArrayRef<Z3_ast> terms) const;
  Z3_ast disjunction(llvm::ArrayRef<Z3_ast> terms) const;
  Z3_ast maximum(llvm::ArrayRef<Z3_ast> terms) const;
  Z3_ast maximum(Z3_ast lhs, Z3_ast rhs) const;
  Z3_ast implies(Z3_ast premise, Z3_ast conclusion) const;

  void assertFormula(Z3_ast formula) const;
  Z3_context getContext() const { return context; }

private:
  Z3_context context;
  Z3_solver solver;
  Z3_sort intSort;
};

} // namespace mlir::triton::gpu

#endif // TRITON_ENABLE_Z3_JOINT_SOLVER

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_Z3_CONSTRAINT_ENCODER_H
