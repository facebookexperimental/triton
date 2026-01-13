//===- NumericalInvariance.cpp - TTGIR Numerical Fingerprinting -*- C++ -*-===//
//
// Part of the Triton project.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/NumericalInvariance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"

#include <sstream>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUPRINTNUMERICALINVARIANCE
#define GEN_PASS_DEF_TRITONGPUDUMPNUMERICALINVARIANCE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// NumericalInvariance Implementation
//===----------------------------------------------------------------------===//

size_t NumericalInvariance::fingerprint() const {
  return llvm::hash_combine(computationDagHash,
                            llvm::hash_value(dtypeSignature), layoutHash,
                            hwConfigHash);
}

bool NumericalInvariance::operator==(const NumericalInvariance &other) const {
  return computationDagHash == other.computationDagHash &&
         dtypeSignature == other.dtypeSignature &&
         layoutHash == other.layoutHash && hwConfigHash == other.hwConfigHash;
}

void NumericalInvariance::print(llvm::raw_ostream &os) const {
  os << "NumericalInvariance {\n";
  os << "  computationDagHash: " << llvm::formatv("{0:x16}", computationDagHash)
     << "\n";
  os << "  dtypeSignature: " << dtypeSignature << "\n";
  os << "  layoutHash: " << llvm::formatv("{0:x16}", layoutHash) << "\n";
  os << "  hwConfigHash: " << llvm::formatv("{0:x16}", hwConfigHash) << "\n";
  os << "  fingerprint: " << llvm::formatv("{0:x16}", fingerprint()) << "\n";
  os << "}\n";
}

void NumericalInvariance::printDiff(llvm::raw_ostream &os,
                                    const NumericalInvariance &lhs,
                                    const NumericalInvariance &rhs) {
  os << "=== Numerical Invariance Diff ===\n";

  if (lhs.computationDagHash != rhs.computationDagHash) {
    os << "  [DIFF] computationDagHash:\n";
    os << "    LHS: " << llvm::formatv("{0:x16}", lhs.computationDagHash)
       << "\n";
    os << "    RHS: " << llvm::formatv("{0:x16}", rhs.computationDagHash)
       << "\n";
  }

  if (lhs.dtypeSignature != rhs.dtypeSignature) {
    os << "  [DIFF] dtypeSignature:\n";
    os << "    LHS: " << lhs.dtypeSignature << "\n";
    os << "    RHS: " << rhs.dtypeSignature << "\n";
  }

  if (lhs.layoutHash != rhs.layoutHash) {
    os << "  [DIFF] layoutHash:\n";
    os << "    LHS: " << llvm::formatv("{0:x16}", lhs.layoutHash) << "\n";
    os << "    RHS: " << llvm::formatv("{0:x16}", rhs.layoutHash) << "\n";
  }

  if (lhs.hwConfigHash != rhs.hwConfigHash) {
    os << "  [DIFF] hwConfigHash:\n";
    os << "    LHS: " << llvm::formatv("{0:x16}", lhs.hwConfigHash) << "\n";
    os << "    RHS: " << llvm::formatv("{0:x16}", rhs.hwConfigHash) << "\n";
  }

  if (lhs == rhs) {
    os << "  [SAME] Invariances are identical\n";
  }
}

//===----------------------------------------------------------------------===//
// Detail Implementation - Type Hashing
//===----------------------------------------------------------------------===//

namespace detail {

size_t hashType(Type type) {
  size_t hash = 0;

  // Hash the type kind/dialect
  hash = llvm::hash_combine(hash, type.getTypeID().getAsOpaquePointer());

  // Special handling for tensor types to include layout encoding
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Hash element type
    hash = llvm::hash_combine(hash, hashType(tensorTy.getElementType()));

    // Hash shape
    for (int64_t dim : tensorTy.getShape()) {
      hash = llvm::hash_combine(hash, dim);
    }

    // Hash encoding (layout) - this is critical for numerical invariance
    if (Attribute encoding = tensorTy.getEncoding()) {
      hash = llvm::hash_combine(hash, hashLayoutEncoding(encoding));
    }
  }
  // Handle MemDescType for shared memory
  else if (auto memDescTy = dyn_cast<MemDescType>(type)) {
    hash = llvm::hash_combine(hash, hashType(memDescTy.getElementType()));

    for (int64_t dim : memDescTy.getShape()) {
      hash = llvm::hash_combine(hash, dim);
    }

    if (Attribute encoding = memDescTy.getEncoding()) {
      hash = llvm::hash_combine(hash, hashLayoutEncoding(encoding));
    }
  }
  // Standard scalar types
  else if (auto floatTy = dyn_cast<FloatType>(type)) {
    hash = llvm::hash_combine(hash, floatTy.getWidth());
  } else if (auto intTy = dyn_cast<IntegerType>(type)) {
    hash = llvm::hash_combine(hash, intTy.getWidth());
    hash = llvm::hash_combine(hash, intTy.getSignedness());
  }

  return hash;
}

size_t hashLayoutEncoding(Attribute encoding) {
  if (!encoding)
    return 0;

  // Use MLIR's built-in attribute hashing as the base
  size_t hash = mlir::hash_value(encoding);

  // For specific encoding types, we can add additional semantic hashing
  // This helps distinguish numerically-significant parameters

  if (auto blockedEnc = dyn_cast<BlockedEncodingAttr>(encoding)) {
    // Hash the blocked encoding parameters
    for (unsigned val : blockedEnc.getSizePerThread()) {
      hash = llvm::hash_combine(hash, val);
    }
    for (unsigned val : blockedEnc.getThreadsPerWarp()) {
      hash = llvm::hash_combine(hash, val);
    }
    for (unsigned val : blockedEnc.getWarpsPerCTA()) {
      hash = llvm::hash_combine(hash, val);
    }
    for (unsigned val : blockedEnc.getOrder()) {
      hash = llvm::hash_combine(hash, val);
    }
  } else if (auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(encoding)) {
    // MMA version is critical for numerical behavior
    hash = llvm::hash_combine(hash, mmaEnc.getVersionMajor());
    hash = llvm::hash_combine(hash, mmaEnc.getVersionMinor());

    for (unsigned val : mmaEnc.getWarpsPerCTA()) {
      hash = llvm::hash_combine(hash, val);
    }

    // Instruction shape affects computation
    for (unsigned val : mmaEnc.getInstrShape()) {
      hash = llvm::hash_combine(hash, val);
    }
  } else if (auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(encoding)) {
    // Swizzling parameters affect memory access patterns
    hash = llvm::hash_combine(hash, sharedEnc.getVec());
    hash = llvm::hash_combine(hash, sharedEnc.getPerPhase());
    hash = llvm::hash_combine(hash, sharedEnc.getMaxPhase());

    for (unsigned val : sharedEnc.getOrder()) {
      hash = llvm::hash_combine(hash, val);
    }
  } else if (auto nvmmaSharedEnc =
                 dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    hash = llvm::hash_combine(hash, nvmmaSharedEnc.getSwizzlingByteWidth());
    hash = llvm::hash_combine(hash, nvmmaSharedEnc.getTransposed());
    hash = llvm::hash_combine(hash, nvmmaSharedEnc.getElementBitWidth());
    hash = llvm::hash_combine(hash, nvmmaSharedEnc.getFp4Padded());
  }

  return hash;
}

std::optional<size_t> hashHWConfig(Operation *op) {
  size_t hash = 0;
  bool hasHWConfig = false;

  // Check for tensor types with MMA/MFMA encodings
  for (Type resultType : op->getResultTypes()) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(resultType)) {
      if (auto encoding = tensorTy.getEncoding()) {
        if (isa<NvidiaMmaEncodingAttr, AMDMfmaEncodingAttr>(encoding)) {
          hash = llvm::hash_combine(hash, hashLayoutEncoding(encoding));
          hasHWConfig = true;
        }
      }
    }
  }

  // Check for shared memory operations with specific encodings
  for (Type operandType : op->getOperandTypes()) {
    if (auto memDescTy = dyn_cast<MemDescType>(operandType)) {
      if (auto encoding = memDescTy.getEncoding()) {
        if (isa<SwizzledSharedEncodingAttr, NVMMASharedEncodingAttr>(
                encoding)) {
          hash = llvm::hash_combine(hash, hashLayoutEncoding(encoding));
          hasHWConfig = true;
        }
      }
    }
  }

  if (hasHWConfig)
    return hash;
  return std::nullopt;
}

size_t hashOperation(Operation *op,
                     const llvm::DenseMap<Operation *, size_t> &opHashes) {
  size_t hash = 0;

  // 1. Hash the operation name (mnemonic)
  hash = llvm::hash_combine(hash, llvm::hash_value(op->getName().getStringRef()));

  // 2. Hash operand types
  for (Type operandType : op->getOperandTypes()) {
    hash = llvm::hash_combine(hash, hashType(operandType));
  }

  // 3. Hash result types
  for (Type resultType : op->getResultTypes()) {
    hash = llvm::hash_combine(hash, hashType(resultType));
  }

  // 4. Hash attributes (sorted by name for determinism)
  SmallVector<NamedAttribute> sortedAttrs(op->getAttrs().begin(),
                                          op->getAttrs().end());
  llvm::sort(sortedAttrs, [](const NamedAttribute &a, const NamedAttribute &b) {
    return a.getName().strref() < b.getName().strref();
  });

  for (const NamedAttribute &attr : sortedAttrs) {
    // Skip debug/location info and naming attributes (not numerically significant)
    StringRef attrName = attr.getName().strref();
    if (attrName == "loc" || attrName == "debug_info" ||
        attrName == "sym_name" || attrName == "sym_visibility" ||
        attrName == "function_type" || attrName == "arg_attrs" ||
        attrName == "res_attrs")
      continue;

    hash = llvm::hash_combine(hash, llvm::hash_value(attr.getName().strref()));
    hash = llvm::hash_combine(hash, mlir::hash_value(attr.getValue()));
  }

  // 5. Hash operand producer hashes (structural hashing)
  // This is what makes us insensitive to SSA value names
  for (Value operand : op->getOperands()) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      auto it = opHashes.find(definingOp);
      if (it != opHashes.end()) {
        hash = llvm::hash_combine(hash, it->second);
      }
    } else {
      // Block argument - hash the argument index
      auto blockArg = cast<BlockArgument>(operand);
      hash = llvm::hash_combine(hash, blockArg.getArgNumber());
    }
  }

  return hash;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Main Fingerprint Computation
//===----------------------------------------------------------------------===//

namespace {

/// Helper function to compute invariance from operations within a region.
void computeInvarianceFromRegion(Region &region, NumericalInvariance &inv,
                                  llvm::DenseMap<Operation *, size_t> &opHashes) {
  region.walk([&](Operation *op) {
    // Skip function operations themselves
    if (isa<func::FuncOp>(op))
      return;

    // Compute and store operation hash
    size_t opHash = detail::hashOperation(op, opHashes);
    opHashes[op] = opHash;

    // Update computation DAG hash
    inv.computationDagHash =
        llvm::hash_combine(inv.computationDagHash, opHash);

    // Update layout hash from result types
    for (Type resultType : op->getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(resultType)) {
        if (Attribute encoding = tensorTy.getEncoding()) {
          inv.layoutHash = llvm::hash_combine(
              inv.layoutHash, detail::hashLayoutEncoding(encoding));
        }
      }
    }

    // Update HW config hash
    if (auto hwHash = detail::hashHWConfig(op)) {
      inv.hwConfigHash = llvm::hash_combine(inv.hwConfigHash, *hwHash);
    }
  });
}

} // namespace

NumericalInvariance computeNumericalInvariance(func::FuncOp func) {
  NumericalInvariance inv;
  llvm::DenseMap<Operation *, size_t> opHashes;

  // Build dtype signature from function signature
  std::ostringstream dtypeSig;

  // Input types
  bool firstInput = true;
  for (Type inputType : func.getArgumentTypes()) {
    if (!firstInput)
      dtypeSig << ",";
    firstInput = false;

    if (auto tensorTy = dyn_cast<RankedTensorType>(inputType)) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      tensorTy.getElementType().print(os);
      dtypeSig << os.str();
    }
  }

  dtypeSig << "->";

  // Output types
  bool firstOutput = true;
  for (Type outputType : func.getResultTypes()) {
    if (!firstOutput)
      dtypeSig << ",";
    firstOutput = false;

    if (auto tensorTy = dyn_cast<RankedTensorType>(outputType)) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      tensorTy.getElementType().print(os);
      dtypeSig << os.str();
    }
  }

  inv.dtypeSignature = dtypeSig.str();

  // Walk operations in the function body
  for (Region &region : func->getRegions()) {
    computeInvarianceFromRegion(region, inv, opHashes);
  }

  return inv;
}

NumericalInvariance computeNumericalInvariance(ModuleOp module) {
  NumericalInvariance inv;
  llvm::DenseMap<Operation *, size_t> opHashes;

  // Build dtype signature from function signatures
  std::ostringstream dtypeSig;
  bool firstFunc = true;

  module.walk([&](func::FuncOp func) {
    if (!firstFunc)
      dtypeSig << ";";
    firstFunc = false;

    // Input types
    bool firstInput = true;
    for (Type inputType : func.getArgumentTypes()) {
      if (!firstInput)
        dtypeSig << ",";
      firstInput = false;

      if (auto tensorTy = dyn_cast<RankedTensorType>(inputType)) {
        std::string typeStr;
        llvm::raw_string_ostream os(typeStr);
        tensorTy.getElementType().print(os);
        dtypeSig << os.str();
      }
    }

    dtypeSig << "->";

    // Output types
    bool firstOutput = true;
    for (Type outputType : func.getResultTypes()) {
      if (!firstOutput)
        dtypeSig << ",";
      firstOutput = false;

      if (auto tensorTy = dyn_cast<RankedTensorType>(outputType)) {
        std::string typeStr;
        llvm::raw_string_ostream os(typeStr);
        tensorTy.getElementType().print(os);
        dtypeSig << os.str();
      }
    }
  });

  inv.dtypeSignature = dtypeSig.str();

  // Walk operations in topological order (natural walk order in MLIR)
  module.walk([&](Operation *op) {
    // Skip module and function operations themselves
    if (isa<ModuleOp>(op) || isa<func::FuncOp>(op))
      return;

    // Compute and store operation hash
    size_t opHash = detail::hashOperation(op, opHashes);
    opHashes[op] = opHash;

    // Update computation DAG hash
    inv.computationDagHash =
        llvm::hash_combine(inv.computationDagHash, opHash);

    // Update layout hash from result types
    for (Type resultType : op->getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(resultType)) {
        if (Attribute encoding = tensorTy.getEncoding()) {
          inv.layoutHash = llvm::hash_combine(
              inv.layoutHash, detail::hashLayoutEncoding(encoding));
        }
      }
    }

    // Update HW config hash
    if (auto hwHash = detail::hashHWConfig(op)) {
      inv.hwConfigHash = llvm::hash_combine(inv.hwConfigHash, *hwHash);
    }
  });

  return inv;
}

size_t computeNumericalFingerprint(ModuleOp module) {
  return computeNumericalInvariance(module).fingerprint();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct PrintNumericalInvariancePass
    : impl::TritonGPUPrintNumericalInvarianceBase<PrintNumericalInvariancePass> {
  using impl::TritonGPUPrintNumericalInvarianceBase<
      PrintNumericalInvariancePass>::TritonGPUPrintNumericalInvarianceBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    NumericalInvariance inv = computeNumericalInvariance(module);
    inv.print(llvm::errs());
  }
};

struct DumpNumericalInvariancePass
    : impl::TritonGPUDumpNumericalInvarianceBase<DumpNumericalInvariancePass> {
  using impl::TritonGPUDumpNumericalInvarianceBase<
      DumpNumericalInvariancePass>::TritonGPUDumpNumericalInvarianceBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    NumericalInvariance inv = computeNumericalInvariance(module);

    if (outputPath.empty()) {
      // If no path specified, print to stderr
      inv.print(llvm::errs());
      return;
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      module.emitError() << "Failed to open file for writing: " << outputPath
                         << " (" << ec.message() << ")";
      signalPassFailure();
      return;
    }

    inv.print(os);
  }
};

std::unique_ptr<Pass> createPrintNumericalInvariancePass() {
  return std::make_unique<PrintNumericalInvariancePass>();
}

std::unique_ptr<Pass> createDumpNumericalInvariancePass(StringRef outputPath) {
  TritonGPUDumpNumericalInvarianceOptions options;
  options.outputPath = outputPath.str();
  return createTritonGPUDumpNumericalInvariance(std::move(options));
}

} // namespace mlir::triton::gpu
