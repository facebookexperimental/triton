#ifndef TLX_INTERFACES
#define TLX_INTERFACES

include "mlir/IR/OpBase.td"

def SameOperandAndResultMemorySpace : NativeOpTrait<"SameOperandAndResultMemorySpace">;

#endif // TLX_INTERFACES
