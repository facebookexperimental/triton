#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonCPUToLLVM : Pass<"convert-triton-cpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonCPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonCPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect"];

    let options = [
        Option<"cpuFeatures", "cpu-features",
               "std::set<std::string>", /*default*/"",
               "comma-separated cpu features">,
    ];
}

#endif
