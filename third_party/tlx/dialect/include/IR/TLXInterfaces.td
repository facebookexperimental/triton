#ifndef TLX_INTERFACES
#define TLX_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def SameOperandAndResultMemorySpace : NativeOpTrait<"SameOperandAndResultMemorySpace">;

#endif // TLX_INTERFACES
