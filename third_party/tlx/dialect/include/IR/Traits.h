#ifndef TRITON_DIALECT_TLX_IR_TRAITS_H_

#define TRITON_DIALECT_TLX_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace impl {

LogicalResult verifySameOperandAndResultMemorySpace(Operation *op);

} // namespace impl

template <typename ConcreteType>
class SameOperandAndResultMemorySpace
    : public TraitBase<ConcreteType, SameOperandAndResultMemorySpace> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandAndResultMemorySpace(op);
  }
};
} // namespace OpTrait
} // namespace mlir

#endif
