#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
class TargetInfo;
// Annotates LocalLoadOps with ttg.amdg.syncedByAsyncWait=true if they are
// synced by an AsyncWait.
void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod);

// Getter for the annotation applied by annotateLocalLoadsSyncedViaAsyncWait
bool isSyncedViaAsyncWait(Operation *localLoadOp);

// LLVM is unable to deduce dependencies across warps and loop iterations for
// AsyncCopy and LocalLoad and will emit conservative wait counts. In triton the
// dependency is models via AsyncWait, e.g.
//   %token1 = ttg.async_copy_global_to_local/amdg.buffer_load_to_local
//   %token2 = ttg.async_wait %token1
//   %1      = ttg.local_load .. token %token2
// For such cases AsyncWait will emit the correct wait and the conservative
// waits are redundant and hindering performance/interleaving.
// To disable the conservative waits two alias scopes are created:
//   1) "amdg.AsyncCopies" will contain all AsyncCopy ops
//   2) "amdg.LocalLoad" will contain all LocalLoads manually synchronized via
//      AsyncWait
// ALl manually synchronized LocalLoads will additionally have "AsyncCopies" as
// a non alias scope to disable the implicit waits from the LLVM backend

// If localLoadOp has a token from an AsyncWait:
//  - Attaches "amdg.LocalLoad" alias scope to llLoadOp
//  - Attaches "amdg.AsyncCopies" as *non* alias scope to llLoadOp
void addLocalLoadNoAliasScope(Operation *localLoadOp,
                              LLVM::AliasAnalysisOpInterface llLoadOp);
// Overload from above without checking the AsyncToken
void addLocalLoadNoAliasScope(LLVM::AliasAnalysisOpInterface llLoadOp);
// Attaches the "AsyncCopies" alias scope to llLoadDirectToLdsOp
void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface llLoadDirectToLdsOp);

// Finds the largest supported vecSize smaller than maxVecSize. Returns 0 if
// there is none
unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::AMD::TargetInfo &targetInfo);

// Derive the producer (global-side) register-tensor LinearLayout that makes a
// direct-to-LDS write into a `padded_shared` LDS layout coalesced, given the
// per-thread load width `loadContig`. `sharedOffsetLayout` is the padded
// encoding's linear component (its "offset" bases map LDS offsets -> n-D tensor
// indices). Offset bases are handed out as: log2(loadContig) register bases,
// log2(threadsPerWarp) lane bases, log2(numWarps) warp bases, then any leftover
// as extra register bases -- so each lane writes `loadContig` consecutive LDS
// elements. Combined with `cgaLayout`/`shape` into a full {reg,lane,warp,block}
// layout. Fails if the LDS shape is too small or the result cannot reach
// `loadContig` consecutive elements.
//
// This is the base-assignment `tritonamdgpu-coalesce-async-copy` applies to the
// async_copy pointer tensor, shared here so `amdgpu.buffer_load_to_local` can
// infer its offset-tensor layout from a pinned padded shared layout.
FailureOr<triton::LinearLayout> deducePaddedDirectToLdsRegLayout(
    const triton::LinearLayout &sharedOffsetLayout, unsigned loadContig,
    unsigned threadsPerWarp, unsigned numWarps, ArrayRef<int64_t> shape,
    triton::gpu::CGAEncodingAttr cgaLayout, MLIRContext *ctx);

} // namespace mlir::triton::AMD

#endif
