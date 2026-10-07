#ifndef NVWS_TRANSFORMS_BUFFER_GROUPS_H_
#define NVWS_TRANSFORMS_BUFFER_GROUPS_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::nvws::buffer {

inline constexpr StringLiteral kBufferIdAttrName = "buffer.id";
inline constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
inline constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";
inline constexpr StringLiteral kBufferCircularAttrName = "buffer.circular";
inline constexpr StringLiteral kBufferStartAttrName = "buffer.start";

std::optional<int64_t> getI64Attr(Operation *op, StringRef name);
bool isSupportedAliasOp(Operation *op);

enum class MemoryKind { Tmem, Local };

// Storage membership only. Consumers attach their own sizing and planning data.
struct Group {
  int64_t bufferId = 0;
  MemoryKind memory = MemoryKind::Tmem;
  bool circular = false;
  SmallVector<Operation *, 2> allocations;
};

// Collect TMEM groups, ordinary mutable local groups, then individual circular
// local allocations, preserving allocation order within each group. A null
// functionBlock collects the whole function; otherwise collect that block and
// its nested operations.
FailureOr<SmallVector<Group, 0>>
collectGroups(FuncOp func, Block *functionBlock = nullptr,
              StringRef diagnosticPrefix = "nvws-insert-semas");

Block *getTopLevelFunctionBlock(Operation *op, FuncOp func);

// Find the unique top-level function CFG block containing the group's complete
// memdesc/token use closure, falling back to its first allocation when unused.
FailureOr<Block *> getManagedGroupUseBlock(const Group &group, FuncOp func,
                                           StringRef diagnosticPrefix);

// Every group and its complete memdesc/token use closure must reside in one
// top-level function CFG block. Group metadata errors retain the collector's
// default diagnostic prefix; locality errors use the caller's prefix.
LogicalResult validateManagedAllocationLocality(FuncOp func,
                                                StringRef diagnosticPrefix);

} // namespace mlir::triton::nvws::buffer

#endif // NVWS_TRANSFORMS_BUFFER_GROUPS_H_
