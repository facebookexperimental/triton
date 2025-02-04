#ifndef TRITON_THIRD_PARTY_CPU_INCLUDE_TRITONCPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_CPU_INCLUDE_TRITONCPUTOLLVM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

}; // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonCPUToLLVMPass(const std::set<std::string> &cpuFeatures = {});

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif
