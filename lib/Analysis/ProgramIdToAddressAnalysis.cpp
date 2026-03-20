//===- ProgramIdToAddressAnalysis.cpp - Program ID to Address Analysis ---===//
//
// Part of the Triton Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/ProgramIdToAddressAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "program-id-to-address-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// ProgramIdAddressInfo Implementation
//===----------------------------------------------------------------------===//

ProgramIdAddressInfo
ProgramIdAddressInfo::getPessimisticValueState(Value value) {
  ProgramIdAddressInfo info;
  info.pattern = Pattern::UNKNOWN;
  return info;
}

ProgramIdAddressInfo
ProgramIdAddressInfo::join(const ProgramIdAddressInfo &lhs,
                           const ProgramIdAddressInfo &rhs) {
  // UNKNOWN is the bottom of the lattice - joining with bottom gives the other
  // This is critical for dataflow analysis initialization
  if (lhs.isUnknown()) {
    return rhs;
  }
  if (rhs.isUnknown()) {
    return lhs;
  }

  // If both are constants, they must match
  if (lhs.isConstant() && rhs.isConstant()) {
    if (lhs.constantOffset == rhs.constantOffset &&
        lhs.hasVariableOffset == rhs.hasVariableOffset) {
      return lhs;
    }
    // Constants differ - could indicate different control flow paths
    // Mark as having variable offset
    ProgramIdAddressInfo result = lhs;
    result.hasVariableOffset = true;
    return result;
  }

  // If patterns don't match, go to pessimistic (top = NONLINEAR)
  if (lhs.pattern != rhs.pattern) {
    ProgramIdAddressInfo result;
    result.pattern = Pattern::PID_NONLINEAR;
    return result;
  }

  // For affine patterns, check if components match
  if (lhs.isPidAffine()) {
    // Check if pid components match
    if (lhs.pidComponents.size() != rhs.pidComponents.size()) {
      ProgramIdAddressInfo result;
      result.pattern = Pattern::PID_NONLINEAR;
      return result;
    }

    for (size_t i = 0; i < lhs.pidComponents.size(); ++i) {
      if (!(lhs.pidComponents[i] == rhs.pidComponents[i])) {
        ProgramIdAddressInfo result;
        result.pattern = Pattern::PID_NONLINEAR;
        return result;
      }
    }

    // If we get here, the affine structures match
    // Return lhs but mark as having variable offset if offsets differ
    ProgramIdAddressInfo result = lhs;
    if (lhs.constantOffset != rhs.constantOffset || lhs.hasVariableOffset ||
        rhs.hasVariableOffset) {
      result.hasVariableOffset = true;
    }
    return result;
  }

  // For NONLINEAR, stay NONLINEAR (top of lattice)
  if (lhs.isPidNonlinear()) {
    return lhs;
  }

  ProgramIdAddressInfo result;
  result.pattern = Pattern::PID_NONLINEAR;
  return result;
}

void ProgramIdAddressInfo::addPidComponent(int axis, int64_t stride) {
  // Check if we already have this axis
  for (auto &comp : pidComponents) {
    if (comp.axis == axis) {
      comp.stride += stride;
      return;
    }
  }
  // Add new component
  pidComponents.push_back({axis, stride});
}

void ProgramIdAddressInfo::multiplyStrides(int64_t factor) {
  for (auto &comp : pidComponents) {
    comp.stride *= factor;
  }
  constantOffset *= factor;
}

void ProgramIdAddressInfo::addConstantOffset(int64_t offset) {
  constantOffset += offset;
}

bool ProgramIdAddressInfo::operator==(const ProgramIdAddressInfo &other) const {
  if (pattern != other.pattern)
    return false;
  if (pattern == Pattern::UNKNOWN)
    return true;
  if (pattern == Pattern::CONSTANT)
    return constantOffset == other.constantOffset;

  // Compare affine patterns
  if (pidComponents.size() != other.pidComponents.size())
    return false;
  for (size_t i = 0; i < pidComponents.size(); ++i) {
    if (!(pidComponents[i] == other.pidComponents[i]))
      return false;
  }
  return constantOffset == other.constantOffset &&
         hasVariableOffset == other.hasVariableOffset &&
         basePtr == other.basePtr && elemSize == other.elemSize;
}

void ProgramIdAddressInfo::print(raw_ostream &os) const {
  switch (pattern) {
  case Pattern::UNKNOWN:
    os << "UNKNOWN";
    return;
  case Pattern::CONSTANT:
    os << "CONSTANT(" << constantOffset << ")";
    return;
  case Pattern::PID_AFFINE:
    os << "PID_AFFINE(";
    for (size_t i = 0; i < pidComponents.size(); ++i) {
      if (i > 0)
        os << " + ";
      os << "pid[" << pidComponents[i].axis << "] * "
         << pidComponents[i].stride;
    }
    if (constantOffset != 0 || hasVariableOffset) {
      os << " + ";
      if (constantOffset != 0)
        os << constantOffset;
      if (hasVariableOffset)
        os << " + var";
    }
    if (basePtr) {
      os << ", base=<ptr>, elem_size=" << elemSize;
    }
    os << ")";
    return;
  case Pattern::PID_NONLINEAR:
    os << "PID_NONLINEAR";
    return;
  }
}

//===----------------------------------------------------------------------===//
// LoadStoreAddressPattern Implementation
//===----------------------------------------------------------------------===//

void LoadStoreAddressPattern::print(raw_ostream &os) const {
  os << "LoadStoreAddressPattern {\n";
  os << "  op: " << (op ? op->getName().getStringRef() : "<null>") << "\n";
  os << "  pattern: ";
  switch (pattern) {
  case Pattern::UNKNOWN:
    os << "UNKNOWN";
    break;
  case Pattern::CONSTANT:
    os << "CONSTANT";
    break;
  case Pattern::PID_AFFINE:
    os << "PID_AFFINE";
    break;
  case Pattern::PID_NONLINEAR:
    os << "PID_NONLINEAR";
    break;
  }
  os << "\n";

  if (!pidAxisStrides.empty()) {
    os << "  pid strides: ";
    for (const auto &[axis, stride] : pidAxisStrides) {
      os << "[axis=" << axis << ", stride=" << stride << "] ";
    }
    os << "\n";
  }

  if (blockSize.has_value()) {
    os << "  block size: " << *blockSize << "\n";
  }

  os << "  coalesced: " << (isCoalesced ? "true" : "false") << "\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// ProgramIdToAddressAnalysis Implementation
//===----------------------------------------------------------------------===//

ProgramIdToAddressAnalysis::ProgramIdToAddressAnalysis(DataFlowSolver &solver)
    : SparseForwardDataFlowAnalysis(solver) {}

void ProgramIdToAddressAnalysis::setToEntryState(Lattice *lattice) {
  propagateIfChanged(
      lattice, lattice->join(ProgramIdAddressInfo::getPessimisticValueState(
                   lattice->getAnchor())));
}

LogicalResult
ProgramIdToAddressAnalysis::visitOperation(Operation *op,
                                           ArrayRef<const Lattice *> operands,
                                           ArrayRef<Lattice *> results) {

  LDBG("Visiting operation: " << *op);

  // Debug: print that we're visiting
  llvm::errs() << "[PID-ADDR] Visiting: " << op->getName() << "\n";

  // Handle GetProgramIdOp
  if (isa<triton::GetProgramIdOp>(op)) {
    visitGetProgramIdOp(op, results);
    return success();
  }

  // Handle MakeRangeOp
  if (isa<triton::MakeRangeOp>(op)) {
    visitMakeRangeOp(op, results);
    return success();
  }

  // Handle arith.constant
  if (isa<arith::ConstantOp>(op)) {
    visitConstantOp(op, results);
    return success();
  }

  // Handle SplatOp
  if (isa<triton::SplatOp>(op)) {
    visitSplatOp(op, operands, results);
    return success();
  }

  // Handle BroadcastOp
  if (isa<triton::BroadcastOp>(op)) {
    visitBroadcastOp(op, operands, results);
    return success();
  }

  // Handle ExpandDimsOp
  if (isa<triton::ExpandDimsOp>(op)) {
    visitExpandDimsOp(op, operands, results);
    return success();
  }

  // Handle arithmetic operations
  if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(op)) {
    visitArithBinaryOp(op, operands, results);
    return success();
  }

  // Handle AddPtrOp
  if (isa<triton::AddPtrOp>(op)) {
    visitAddPtrOp(op, operands, results);
    return success();
  }

  // Handle LoadOp and StoreOp - just record the pattern
  if (isa<triton::LoadOp, triton::StoreOp>(op)) {
    visitLoadStoreOp(op, operands);
    // Load/store don't produce values we need to track (for store)
    // For load, we could track if needed, but typically we want to know
    // about the address, not the loaded value
    for (auto *result : results) {
      setToEntryState(result);
    }
    return success();
  }

  // Default: set to unknown
  for (auto *result : results) {
    setToEntryState(result);
  }
  return success();
}

void ProgramIdToAddressAnalysis::visitGetProgramIdOp(
    Operation *op, ArrayRef<Lattice *> results) {
  auto getPidOp = cast<triton::GetProgramIdOp>(op);
  int axis = static_cast<int>(getPidOp.getAxisAsInt());

  ProgramIdAddressInfo info = ProgramIdAddressInfo::getProgramId(axis);

  propagateIfChanged(results[0], results[0]->join(info));

  LDBG("  GetProgramIdOp axis=" << axis << " -> " << info);
}

void ProgramIdToAddressAnalysis::visitMakeRangeOp(Operation *op,
                                                  ArrayRef<Lattice *> results) {
  auto rangeOp = cast<triton::MakeRangeOp>(op);

  // make_range produces [start, start+1, ..., end-1]
  // This is a constant pattern with variable elements (0 to end-start-1)
  ProgramIdAddressInfo info;
  info = ProgramIdAddressInfo::getConstant(rangeOp.getStart());
  info.setHasVariableOffset(); // Elements vary within the range

  propagateIfChanged(results[0], results[0]->join(info));

  LDBG("  MakeRangeOp -> " << info);
}

void ProgramIdToAddressAnalysis::visitConstantOp(Operation *op,
                                                 ArrayRef<Lattice *> results) {
  auto constOp = cast<arith::ConstantOp>(op);

  ProgramIdAddressInfo info;

  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
    info = ProgramIdAddressInfo::getConstant(intAttr.getInt());
  } else if (auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue())) {
    if (splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.getSplatValue<APInt>().getSExtValue();
      info = ProgramIdAddressInfo::getConstant(value);
    }
  }

  propagateIfChanged(results[0], results[0]->join(info));

  LDBG("  ConstantOp -> " << info);
}

void ProgramIdToAddressAnalysis::visitSplatOp(
    Operation *op, ArrayRef<const Lattice *> operands,
    ArrayRef<Lattice *> results) {
  // Splat broadcasts a scalar to a tensor - preserves the address info
  if (!operands.empty()) {
    ProgramIdAddressInfo info = operands[0]->getValue();
    propagateIfChanged(results[0], results[0]->join(info));
    LDBG("  SplatOp -> " << info);
  }
}

void ProgramIdToAddressAnalysis::visitBroadcastOp(
    Operation *op, ArrayRef<const Lattice *> operands,
    ArrayRef<Lattice *> results) {
  // Broadcast expands a tensor - preserves the address info
  if (!operands.empty()) {
    ProgramIdAddressInfo info = operands[0]->getValue();
    propagateIfChanged(results[0], results[0]->join(info));
    LDBG("  BroadcastOp -> " << info);
  }
}

void ProgramIdToAddressAnalysis::visitExpandDimsOp(
    Operation *op, ArrayRef<const Lattice *> operands,
    ArrayRef<Lattice *> results) {
  // ExpandDims adds a dimension - preserves the address info
  if (!operands.empty()) {
    ProgramIdAddressInfo info = operands[0]->getValue();
    propagateIfChanged(results[0], results[0]->join(info));
    LDBG("  ExpandDimsOp -> " << info);
  }
}

void ProgramIdToAddressAnalysis::visitArithBinaryOp(
    Operation *op, ArrayRef<const Lattice *> operands,
    ArrayRef<Lattice *> results) {
  if (operands.size() != 2) {
    setToEntryState(results[0]);
    return;
  }

  const ProgramIdAddressInfo &lhs = operands[0]->getValue();
  const ProgramIdAddressInfo &rhs = operands[1]->getValue();
  ProgramIdAddressInfo result;

  if (isa<arith::MulIOp>(op)) {
    // Multiplication: pid * constant or constant * pid
    if (lhs.isPidAffine() && rhs.isConstant()) {
      result = lhs;
      result.multiplyStrides(rhs.getConstantOffset());
    } else if (lhs.isConstant() && rhs.isPidAffine()) {
      result = rhs;
      result.multiplyStrides(lhs.getConstantOffset());
    } else if (lhs.isConstant() && rhs.isConstant()) {
      result = ProgramIdAddressInfo::getConstant(lhs.getConstantOffset() *
                                                 rhs.getConstantOffset());
    } else if (lhs.isPidAffine() && rhs.isPidAffine()) {
      // pid * pid -> nonlinear
      result.pattern = ProgramIdAddressInfo::Pattern::PID_NONLINEAR;
    }
  } else if (isa<arith::AddIOp>(op)) {
    // Addition: combine pid components and offsets
    if (lhs.isConstant() && rhs.isConstant()) {
      result = ProgramIdAddressInfo::getConstant(lhs.getConstantOffset() +
                                                 rhs.getConstantOffset());
    } else if (lhs.isPidAffine() && rhs.isConstant()) {
      result = lhs;
      result.addConstantOffset(rhs.getConstantOffset());
      if (rhs.getHasVariableOffset())
        result.setHasVariableOffset();
    } else if (lhs.isConstant() && rhs.isPidAffine()) {
      result = rhs;
      result.addConstantOffset(lhs.getConstantOffset());
      if (lhs.getHasVariableOffset())
        result.setHasVariableOffset();
    } else if (lhs.isPidAffine() && rhs.isPidAffine()) {
      // Combine two affine expressions
      result = lhs;
      for (const auto &comp : rhs.getPidComponents()) {
        result.addPidComponent(comp.axis, comp.stride);
      }
      result.addConstantOffset(rhs.getConstantOffset());
      if (rhs.getHasVariableOffset())
        result.setHasVariableOffset();
    } else if (lhs.isPidAffine()) {
      result = lhs;
      result.setHasVariableOffset();
    } else if (rhs.isPidAffine()) {
      result = rhs;
      result.setHasVariableOffset();
    }
  } else if (isa<arith::SubIOp>(op)) {
    // Subtraction: similar to addition but negate rhs
    if (lhs.isConstant() && rhs.isConstant()) {
      result = ProgramIdAddressInfo::getConstant(lhs.getConstantOffset() -
                                                 rhs.getConstantOffset());
    } else if (lhs.isPidAffine() && rhs.isConstant()) {
      result = lhs;
      result.addConstantOffset(-rhs.getConstantOffset());
    } else if (lhs.isPidAffine()) {
      result = lhs;
      result.setHasVariableOffset();
    }
  }

  propagateIfChanged(results[0], results[0]->join(result));
  LDBG("  ArithBinaryOp " << op->getName() << " -> " << result);
}

void ProgramIdToAddressAnalysis::visitAddPtrOp(
    Operation *op, ArrayRef<const Lattice *> operands,
    ArrayRef<Lattice *> results) {
  if (operands.size() != 2) {
    setToEntryState(results[0]);
    return;
  }

  auto addPtrOp = cast<triton::AddPtrOp>(op);
  const ProgramIdAddressInfo &ptrInfo = operands[0]->getValue();
  const ProgramIdAddressInfo &offsetInfo = operands[1]->getValue();

  // Get element size
  int64_t elemSize = std::max<int64_t>(
      1, triton::getPointeeBitWidth(addPtrOp.getPtr().getType()) / 8);

  ProgramIdAddressInfo result;

  // The offset is scaled by element size
  if (offsetInfo.isPidAffine() || offsetInfo.isConstant()) {
    result = offsetInfo;
    result.multiplyStrides(elemSize);
    result.setBasePtr(addPtrOp.getPtr(), elemSize);

    // Inherit base ptr from ptr operand if it has one
    if (ptrInfo.getBasePtr()) {
      result.setBasePtr(ptrInfo.getBasePtr(), elemSize);
    }
  } else {
    result = ProgramIdAddressInfo::getPessimisticValueState(op->getResult(0));
  }

  propagateIfChanged(results[0], results[0]->join(result));
  LDBG("  AddPtrOp (elemSize=" << elemSize << ") -> " << result);
}

void ProgramIdToAddressAnalysis::visitLoadStoreOp(
    Operation *op, ArrayRef<const Lattice *> operands) {
  if (operands.empty())
    return;

  const ProgramIdAddressInfo &ptrInfo = operands[0]->getValue();

  LoadStoreAddressPattern pattern;
  pattern.op = op;
  pattern.pattern = ptrInfo.getPattern();
  pattern.basePtr = ptrInfo.getBasePtr();

  if (ptrInfo.isPidAffine()) {
    for (const auto &comp : ptrInfo.getPidComponents()) {
      pattern.pidAxisStrides.emplace_back(comp.axis, comp.stride);
    }

    // Estimate block size from the variable offset range
    // This is a simplification - in practice we'd need more analysis
    if (!ptrInfo.getHasVariableOffset()) {
      pattern.blockSize = 1;
    }

    // Simple coalescing heuristic: check if stride is contiguous for axis 0
    // (In practice, this needs more sophisticated analysis)
    pattern.isCoalesced =
        !ptrInfo.getPidComponents().empty() && ptrInfo.getElemSize() > 0;
  }

  loadStorePatterns[op] = pattern;

  LDBG("  LoadStoreOp recorded pattern: ");
  LLVM_DEBUG(pattern.print(llvm::dbgs()));
}

LoadStoreAddressPattern
ProgramIdToAddressAnalysis::getAddressPattern(Operation *loadOrStore) {
  auto it = loadStorePatterns.find(loadOrStore);
  if (it != loadStorePatterns.end()) {
    return it->second;
  }
  return LoadStoreAddressPattern{};
}

//===----------------------------------------------------------------------===//
// ModuleProgramIdToAddressAnalysis Implementation
//===----------------------------------------------------------------------===//

ModuleProgramIdToAddressAnalysis::ModuleProgramIdToAddressAnalysis(
    ModuleOp moduleOp) {
  // Create and run the dataflow solver internally (like ModuleAxisInfoAnalysis)
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<ProgramIdToAddressAnalysis>();

  if (failed(solver.initializeAndRun(moduleOp))) {
    return;
  }

  // Collect all value info and load/store patterns by walking the module
  moduleOp.walk([&](triton::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Store info for all values
      for (Value result : op->getResults()) {
        auto *lattice =
            solver.lookupState<dataflow::Lattice<ProgramIdAddressInfo>>(result);
        if (lattice) {
          valueInfoMap[result] = lattice->getValue();
        }
      }

      // Record load/store patterns
      if (isa<triton::LoadOp, triton::StoreOp>(op)) {
        Value ptrOperand = op->getOperand(0);
        auto *lattice =
            solver.lookupState<dataflow::Lattice<ProgramIdAddressInfo>>(
                ptrOperand);
        if (!lattice)
          return;

        const ProgramIdAddressInfo &ptrInfo = lattice->getValue();

        LoadStoreAddressPattern pattern;
        pattern.op = op;
        pattern.pattern = ptrInfo.getPattern();
        pattern.basePtr = ptrInfo.getBasePtr();

        if (ptrInfo.isPidAffine()) {
          for (const auto &comp : ptrInfo.getPidComponents()) {
            pattern.pidAxisStrides.emplace_back(comp.axis, comp.stride);
          }

          if (!ptrInfo.getHasVariableOffset()) {
            pattern.blockSize = 1;
          }

          pattern.isCoalesced =
              !ptrInfo.getPidComponents().empty() && ptrInfo.getElemSize() > 0;
        }

        patterns[op] = pattern;
      }
    });
  });
}

const LoadStoreAddressPattern *
ModuleProgramIdToAddressAnalysis::getAddressPattern(
    Operation *loadOrStore) const {
  auto it = patterns.find(loadOrStore);
  if (it != patterns.end()) {
    return &it->second;
  }
  return nullptr;
}

const ProgramIdAddressInfo *
ModuleProgramIdToAddressAnalysis::getInfo(Value value) const {
  auto it = valueInfoMap.find(value);
  if (it != valueInfoMap.end()) {
    return &it->second;
  }
  return nullptr;
}

void ModuleProgramIdToAddressAnalysis::print(raw_ostream &os) const {
  os << "=== Program ID to Address Analysis Results ===\n";
  for (const auto &[op, pattern] : patterns) {
    os << "\nOperation at " << op->getLoc() << ":\n";
    pattern.print(os);
  }
}

} // namespace mlir::triton
