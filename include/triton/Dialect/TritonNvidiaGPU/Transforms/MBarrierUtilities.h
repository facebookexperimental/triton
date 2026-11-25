#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace mlir::triton::nvidia_gpu {

bool isCrossCTALoadStore(::mlir::triton::gpu::MemDescType memDescTy,
                         ::mlir::RankedTensorType regTy);

bool isCrossCTAGatherScatter(::mlir::triton::gpu::MemDescType memDescTy,
                             ::mlir::RankedTensorType regTy, unsigned axis);

// Checks shared-memory payload accesses, including allocation initializers.
// Returns nullopt for unsupported accesses or unknown CTA routing.
std::optional<bool> hasCrossCTASharedAccess(Operation *op);

bool hasTCGen5CommitCrossCTA(Operation *op);

// Checks routing of explicit mbarrier operands; callers check barrier-layout
// broadcast separately. Requires the module's two-CTA mode to be resolved.
bool hasCrossCTAMBarrierUse(gpu::MBarrierOpInterface op);

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier);

// Whether `op` is a per-thread arrival that every thread of its region
// performs. A predicate can leave threads out, and the arrival does not
// publish their effects.
bool arrivesFromEveryThread(Operation *op);

// Whether `op` is a named barrier whose thread count includes every thread
// of its region, so it completes only after all of them reach it, if they
// all execute it.
bool countsEveryThread(Operation *op);

} // namespace mlir::triton::nvidia_gpu
