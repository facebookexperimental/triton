#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_NAMEDBARRIER_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_NAMEDBARRIER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::triton::nvidia_gpu {

inline constexpr int32_t kDefaultWarpGroupBarrierId = 0;
inline constexpr int32_t kSwitchLoopBarrierId = 1;
inline constexpr int32_t kFirstPartitionBarrierId = 2;
inline constexpr int32_t kFirstAllocatableBarrierId = 3;
inline constexpr int32_t kLastNamedBarrierId = 15;
inline constexpr llvm::StringLiteral kWarpSpecializeBarrierIdsAttrName =
    "ttng.warp_specialize_barrier_ids";

class NamedBarrierIdAllocator {
public:
  explicit NamedBarrierIdAllocator(ModuleOp module);

  bool hasDynamicUserId() const { return hasDynamicUserId_; }
  bool isReservedByUser(int32_t id) const;
  bool isReservedByCompiler(int32_t id) const;
  std::optional<SmallVector<int32_t>> allocate(unsigned count);
  void reserve(ArrayRef<int32_t> ids);

private:
  std::array<bool, kLastNamedBarrierId + 1> used_{};
  std::array<bool, kLastNamedBarrierId + 1> userIds_{};
  std::array<bool, kLastNamedBarrierId + 1> compilerIds_{};
  bool hasDynamicUserId_ = false;
};

// Assign named-barrier IDs for every warp-specialize partition and record them
// on the module. Use this from a *required* lowering, which must report why it
// cannot proceed.
LogicalResult
ensureWarpSpecializeBarrierIds(ModuleOp module,
                               NamedBarrierIdAllocator &allocator);

// As above, but reports failure without emitting a diagnostic. Use this from an
// *optional* optimization, which must decline silently rather than fail the
// compile when no ID can be proven free -- see `named_barrier_api_changes.md`
// section 4.1.
LogicalResult
tryEnsureWarpSpecializeBarrierIds(ModuleOp module,
                                  NamedBarrierIdAllocator &allocator);
SmallVector<int32_t> getWarpSpecializeBarrierIds(ModuleOp module);
Value createCompilerNamedBarrierId(OpBuilder &builder, Location loc,
                                   int32_t id);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_NAMEDBARRIER_H_
