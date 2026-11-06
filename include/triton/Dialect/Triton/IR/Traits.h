#ifndef TRITON_IR_TRAITS_H_
#define TRITON_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include <atomic>
#include <chrono>
#include <cstdio>

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace impl {
// The rationale for this trait is to prevent users from creating programs
// that would have catastrophic register pressure and cause the compiler to
// hang.
// Since H100 has 256KB registers, we should allow users to create tensors
// of size up to 256K elements. It will spill for datatypes wider than 1B,
// but we probably should limit number of elements (rather than bytes) to
// keep specs simple
int constexpr maxTensorNumElements = 1048576;

LogicalResult verifyTensorSize(Operation *op);
LogicalResult verifyTensorLayouts(Operation *op);

LogicalResult verifySameOperandsEncoding(Operation *op);
LogicalResult verifySameResultsType(Operation *op);
LogicalResult verifyEquivalentTensorType(Type typeA, Type typeB);
LogicalResult verifySameOperandsAndResultEncoding(Operation *op);

LogicalResult verifySameLoadStoreOperandsShape(Operation *op);

LogicalResult verifySameLoadStoreOperandsAndResultShape(Operation *op);

// NOT FOR COMMIT: counts and times the tensor-layout verifier, printed at exit.
struct LayoutVerifyStats {
  std::atomic<uint64_t> traitCalls{0}, traitNs{0}, checkCalls{0}, checkNs{0};
  std::atomic<uint64_t> llCalls{0}, llNs{0};
  ~LayoutVerifyStats() {
    if (traitCalls)
      fprintf(stderr,
              "[layout-verify] verifyTensorLayouts: %llu calls, %.3f ms; "
              "TritonGPU verifyTensorLayout: %llu calls, %.3f ms; "
              "toLinearLayout: %llu calls, %.3f ms\n",
              (unsigned long long)traitCalls.load(), traitNs.load() / 1e6,
              (unsigned long long)checkCalls.load(), checkNs.load() / 1e6,
              (unsigned long long)llCalls.load(), llNs.load() / 1e6);
  }
};
inline LayoutVerifyStats layoutVerifyStats;

struct ScopedNs {
  std::atomic<uint64_t> &ns;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  ~ScopedNs() {
    ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
  }
};

} // namespace impl

template <class ConcreteType>
class TensorSizeTrait : public TraitBase<ConcreteType, TensorSizeTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorSize(op);
  }
};

// Trait applied to all Triton MLIR ops.  Checks that the layouts of tensors are
// valid.
template <class ConcreteType>
class VerifyTensorLayoutsTrait
    : public TraitBase<ConcreteType, VerifyTensorLayoutsTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    impl::layoutVerifyStats.traitCalls++;
    impl::ScopedNs timer{impl::layoutVerifyStats.traitNs};
    return impl::verifyTensorLayouts(op);
  }
};

template <typename ConcreteType>
class SameOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(op);
  }
};

template <typename ConcreteType>
class SameOperandsEncoding
    : public TraitBase<ConcreteType, SameOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op);
  }
};

template <typename ConcreteType>
class SameResultsType : public TraitBase<ConcreteType, SameResultsType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameResultsType(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsAndResultShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(op);
  }
};

// This trait indicates that regions in the op may execute concurrently with
// each other.
template <typename ConcreteType>
struct AsyncRegions : public TraitBase<ConcreteType, AsyncRegions> {};

// This trait marks a one-operand, one-result operation that preserves the
// value's AxisInfo unchanged.
template <typename ConcreteType>
struct AxisInfoPassthroughTrait
    : public TraitBase<ConcreteType, AxisInfoPassthroughTrait> {};

} // namespace OpTrait
} // namespace mlir

#endif
