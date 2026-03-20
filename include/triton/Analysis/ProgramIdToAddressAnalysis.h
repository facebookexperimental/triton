//===- ProgramIdToAddressAnalysis.h - Program ID to Address Analysis -----===//
//
// Part of the Triton Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines an analysis that tracks how load/store addresses relate
// to program_id in Triton programs. This is useful for:
// - Memory coalescing verification
// - Vectorization hints
// - Alias analysis between different program instances
// - Out-of-bounds detection
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_PROGRAMIDTOADDRESSANALYSIS_H
#define TRITON_ANALYSIS_PROGRAMIDTOADDRESSANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// ProgramIdAddressInfo - Lattice value tracking pid-to-address relationship
//===----------------------------------------------------------------------===//

/// Represents how a value relates to program_id in address calculations.
/// The general form tracked is:
///   value = sum(pid[axis] * stride[axis]) + constant_offset + variable_offset
///
/// For pointer values, this becomes:
///   ptr = base_ptr + (sum(pid[axis] * stride[axis]) + offset) * elem_size
class ProgramIdAddressInfo {
public:
  /// Represents a single program_id axis contribution
  struct PidComponent {
    int axis;       // 0, 1, or 2 for x, y, z
    int64_t stride; // Multiplier for this pid axis

    bool operator==(const PidComponent &other) const {
      return axis == other.axis && stride == other.stride;
    }
  };

  /// Classification of the address pattern
  enum class Pattern {
    UNKNOWN,       // Cannot determine relationship
    CONSTANT,      // No dependence on pid (constant across programs)
    PID_AFFINE,    // Linear dependence: base + sum(pid * stride) + offset
    PID_NONLINEAR, // Non-linear dependence (e.g., pid * pid)
  };

public:
  ProgramIdAddressInfo() = default;

  /// Create info for a constant value
  static ProgramIdAddressInfo getConstant(int64_t value) {
    ProgramIdAddressInfo info;
    info.pattern = Pattern::CONSTANT;
    info.constantOffset = value;
    return info;
  }

  /// Create info for a program_id value
  static ProgramIdAddressInfo getProgramId(int axis) {
    ProgramIdAddressInfo info;
    info.pattern = Pattern::PID_AFFINE;
    info.pidComponents.push_back({axis, 1});
    return info;
  }

  /// Create pessimistic (unknown) state
  static ProgramIdAddressInfo getPessimisticValueState(Value value);

  /// Join two lattice values (meet operation for forward analysis)
  static ProgramIdAddressInfo join(const ProgramIdAddressInfo &lhs,
                                   const ProgramIdAddressInfo &rhs);

  // Accessors
  Pattern getPattern() const { return pattern; }
  bool isUnknown() const { return pattern == Pattern::UNKNOWN; }
  bool isConstant() const { return pattern == Pattern::CONSTANT; }
  bool isPidAffine() const { return pattern == Pattern::PID_AFFINE; }
  bool isPidNonlinear() const { return pattern == Pattern::PID_NONLINEAR; }

  ArrayRef<PidComponent> getPidComponents() const { return pidComponents; }
  int64_t getConstantOffset() const { return constantOffset; }
  bool getHasVariableOffset() const { return hasVariableOffset; }
  Value getBasePtr() const { return basePtr; }
  int64_t getElemSize() const { return elemSize; }

  // Modifiers for building address info through operations
  void setBasePtr(Value ptr, int64_t size) {
    basePtr = ptr;
    elemSize = size;
  }

  void addPidComponent(int axis, int64_t stride);
  void multiplyStrides(int64_t factor);
  void addConstantOffset(int64_t offset);
  void setHasVariableOffset() { hasVariableOffset = true; }

  // Comparison
  bool operator==(const ProgramIdAddressInfo &other) const;

  // Debug printing
  void print(raw_ostream &os) const;

private:
  Pattern pattern = Pattern::UNKNOWN;

  // For PID_AFFINE pattern:
  SmallVector<PidComponent, 3> pidComponents; // Up to 3 axes
  int64_t constantOffset = 0;                 // Known constant offset
  bool hasVariableOffset = false; // True if offset has non-constant part

  // For pointer values:
  Value basePtr = nullptr; // Base pointer (if tracking ptr)
  int64_t elemSize = 1;    // Element size for ptr arithmetic

  // Allow access from the analysis
  friend class ProgramIdToAddressAnalysis;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const ProgramIdAddressInfo &info) {
  info.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// LoadStoreAddressPattern - Summary of address pattern for a memory operation
//===----------------------------------------------------------------------===//

/// Summary information about a load/store's address relationship to program_id
struct LoadStoreAddressPattern {
  using Pattern = ProgramIdAddressInfo::Pattern;

  Operation *op = nullptr;            // The load/store operation
  Pattern pattern = Pattern::UNKNOWN; // Address pattern type
  Value basePtr = nullptr;            // Base pointer value

  // For PID_AFFINE pattern:
  // Maps pid axis -> stride (bytes between consecutive program's first element)
  SmallVector<std::pair<int, int64_t>, 3> pidAxisStrides;

  // Block size (number of contiguous elements per program)
  std::optional<int64_t> blockSize;

  // Whether the access pattern is coalesced within a warp
  bool isCoalesced = false;

  void print(raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// ProgramIdToAddressAnalysis - The main analysis pass
//===----------------------------------------------------------------------===//

/// Forward dataflow analysis tracking program_id to address relationships
class ProgramIdToAddressAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<ProgramIdAddressInfo>> {
public:
  using Lattice = dataflow::Lattice<ProgramIdAddressInfo>;

  explicit ProgramIdToAddressAnalysis(DataFlowSolver &solver);

  using dataflow::SparseForwardDataFlowAnalysis<Lattice>::getLatticeElement;

  /// Get the address pattern for a load/store operation
  LoadStoreAddressPattern getAddressPattern(Operation *loadOrStore);

protected:
  void setToEntryState(Lattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const Lattice *> operands,
                               ArrayRef<Lattice *> results) override;

private:
  // Handlers for specific operations
  void visitGetProgramIdOp(Operation *op, ArrayRef<Lattice *> results);
  void visitMakeRangeOp(Operation *op, ArrayRef<Lattice *> results);
  void visitConstantOp(Operation *op, ArrayRef<Lattice *> results);
  void visitSplatOp(Operation *op, ArrayRef<const Lattice *> operands,
                    ArrayRef<Lattice *> results);
  void visitBroadcastOp(Operation *op, ArrayRef<const Lattice *> operands,
                        ArrayRef<Lattice *> results);
  void visitExpandDimsOp(Operation *op, ArrayRef<const Lattice *> operands,
                         ArrayRef<Lattice *> results);
  void visitArithBinaryOp(Operation *op, ArrayRef<const Lattice *> operands,
                          ArrayRef<Lattice *> results);
  void visitAddPtrOp(Operation *op, ArrayRef<const Lattice *> operands,
                     ArrayRef<Lattice *> results);
  void visitLoadStoreOp(Operation *op, ArrayRef<const Lattice *> operands);

  // Cached results for load/store operations
  DenseMap<Operation *, LoadStoreAddressPattern> loadStorePatterns;
};

//===----------------------------------------------------------------------===//
// ModuleProgramIdToAddressAnalysis - Module-level analysis wrapper
//===----------------------------------------------------------------------===//

/// Module-level analysis that runs ProgramIdToAddressAnalysis on all functions
/// This is designed to be used similarly to ModuleAxisInfoAnalysis:
///   ModuleProgramIdToAddressAnalysis analysis(moduleOp);
///   auto* pattern = analysis.getAddressPattern(loadOp);
class ModuleProgramIdToAddressAnalysis {
public:
  explicit ModuleProgramIdToAddressAnalysis(ModuleOp moduleOp);

  /// Get address pattern for a load/store operation
  const LoadStoreAddressPattern *
  getAddressPattern(Operation *loadOrStore) const;

  /// Get the ProgramIdAddressInfo for a value
  const ProgramIdAddressInfo *getInfo(Value value) const;

  /// Get all analyzed load/store patterns
  const DenseMap<Operation *, LoadStoreAddressPattern> &getAllPatterns() const {
    return patterns;
  }

  /// Print all analysis results
  void print(raw_ostream &os) const;

private:
  DenseMap<Operation *, LoadStoreAddressPattern> patterns;
  DenseMap<Value, ProgramIdAddressInfo> valueInfoMap;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_PROGRAMIDTOADDRESSANALYSIS_H
