#ifndef TRITON_THIRD_PARTY_CPU_LIB_TRITONCPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_CPU_LIB_TRITONCPUTOLLVM_UTILITY_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::CPU {

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

} // namespace mlir::LLVM::CPU

#endif // TRITON_THIRD_PARTY_CPU_LIB_TRITONCPUTOLLVM_UTILITY_H_
