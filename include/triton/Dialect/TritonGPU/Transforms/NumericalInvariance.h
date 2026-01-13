//===- NumericalInvariance.h - TTGIR Numerical Fingerprinting ---*- C++ -*-===//
//
// Part of the Triton project.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for computing a "numerical invariance" fingerprint
// for TTGIR modules. Two TTGIR modules with the same fingerprint should produce
// bitwise-identical numerical outputs for the same inputs.
//
// Use case: Regression prevention for numerical mismatches.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_NUMERICALINVARIANCE_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_NUMERICALINVARIANCE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace mlir::triton::gpu {

//===----------------------------------------------------------------------===//
// NumericalInvariance
//===----------------------------------------------------------------------===//

/// Represents the numerical invariance of a TTGIR module.
/// Two modules with the same NumericalInvariance should produce identical
/// numerical outputs for the same inputs.
struct NumericalInvariance {
  /// Hash of the computation DAG structure.
  /// Captures operation types, data flow, but ignores SSA value names.
  size_t computationDagHash = 0;

  /// String signature of data types (inputs, outputs, intermediates).
  /// Format: "input0:f16,input1:f32->output0:f32"
  std::string dtypeSignature;

  /// Hash of all layout encodings in the module.
  /// Includes blocked, mma, shared encodings and their parameters.
  size_t layoutHash = 0;

  /// Hash of hardware-specific configurations.
  /// MMA version, instruction shapes, swizzling parameters, etc.
  size_t hwConfigHash = 0;

  /// Compute the combined fingerprint.
  size_t fingerprint() const;

  /// Compare two invariances for equality.
  bool operator==(const NumericalInvariance &other) const;
  bool operator!=(const NumericalInvariance &other) const {
    return !(*this == other);
  }

  /// Print a human-readable representation for debugging.
  void print(llvm::raw_ostream &os) const;

  /// Print a detailed diff between two invariances.
  static void printDiff(llvm::raw_ostream &os, const NumericalInvariance &lhs,
                        const NumericalInvariance &rhs);
};

//===----------------------------------------------------------------------===//
// Fingerprint Computation
//===----------------------------------------------------------------------===//

/// Compute the numerical invariance for a TTGIR module.
NumericalInvariance computeNumericalInvariance(ModuleOp module);

/// Compute the numerical invariance for a single function.
NumericalInvariance computeNumericalInvariance(mlir::func::FuncOp func);

/// Compute the fingerprint (single hash) for a TTGIR module.
/// This is a convenience wrapper around computeNumericalInvariance().
size_t computeNumericalFingerprint(ModuleOp module);

//===----------------------------------------------------------------------===//
// Operation Hashing (Internal)
//===----------------------------------------------------------------------===//

namespace detail {

/// Hash a single operation, incorporating:
/// - Operation name (mnemonic)
/// - Operand types (with layout encodings)
/// - Result types (with layout encodings)
/// - Attributes (sorted by name)
/// - Hashes of operand-defining operations (structural hashing)
size_t hashOperation(Operation *op,
                     const llvm::DenseMap<Operation *, size_t> &opHashes);

/// Hash a type, including layout encoding for tensor types.
size_t hashType(Type type);

/// Hash a layout encoding attribute.
size_t hashLayoutEncoding(Attribute encoding);

/// Extract and hash hardware-specific configuration from an operation.
/// Returns std::nullopt if the operation has no HW-specific config.
std::optional<size_t> hashHWConfig(Operation *op);

} // namespace detail

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Create a pass that computes and prints the numerical invariance.
std::unique_ptr<Pass> createPrintNumericalInvariancePass();

/// Create a pass that computes and stores the numerical invariance to a file.
std::unique_ptr<Pass> createDumpNumericalInvariancePass(StringRef outputPath);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_NUMERICALINVARIANCE_H_
