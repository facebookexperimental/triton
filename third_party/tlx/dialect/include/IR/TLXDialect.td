#ifndef TLX_DIALECT
#define TLX_DIALECT

include "mlir/IR/OpBase.td"

def TLX_Dialect : Dialect {
  let name = "tlx";
  let cppNamespace = "::mlir::triton::tlx";

  let description = [{
    TLX Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

include "TLXTypes.td"

#endif
