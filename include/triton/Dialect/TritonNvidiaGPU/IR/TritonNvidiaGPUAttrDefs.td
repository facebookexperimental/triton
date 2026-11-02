#ifndef TRITONNVIDIAGPU_ATTRDEFS
#define TRITONNVIDIAGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonAttrEnums.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUEnums.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td"

class TTNG_DefaultCacheModifierParameter
    : EnumParameter<TT_CacheModifierAttr> {
  let defaultValue = "::mlir::triton::CacheModifier::NONE";
}

class TTNG_DefaultCacheEvictionPriorityParameter
    : EnumParameter<TTNG_CacheEvictionPriorityAttr> {
  let defaultValue =
      "::mlir::triton::nvidia_gpu::CacheEvictionPriority::NONE";
}

def TTNG_CachePolicyAttr
    : AttrDef<TritonNvidiaGPU_Dialect, "CachePolicy"> {
  let mnemonic = "cache_policy";
  let description = [{
    NVIDIA global-memory cache hints.

    The attribute can contain a cache modifier, an L1 eviction priority, an L2
    fractional eviction policy, and an L2 prefetch size. This attribute is a
    performance hint and has no memory-ordering semantics.
  }];
  let parameters = (ins
    TTNG_DefaultCacheModifierParameter<>:$cache_modifier,
    TTNG_DefaultCacheEvictionPriorityParameter<>:$l1,
    TTNG_DefaultCacheEvictionPriorityParameter<>:$l2_primary,
    TTNG_DefaultCacheEvictionPriorityParameter<>:$l2_secondary,
    OptionalParameter<"FloatAttr">:$l2_fraction,
    OptionalParameter<"IntegerAttr">:$l2_prefetch_size
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTG_SharedClusterMemorySpace
    : AttrDef<TritonNvidiaGPU_Dialect, "SharedClusterMemorySpace"> {
  let mnemonic = "shared_cluster_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory. The shared memory could reside in
    any CTA within a CTA cluster.
  }];
}

def TTG_TensorMemorySpace
    : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemorySpace"> {
  let mnemonic = "tensor_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to tensor memory.
    The memory is laid out in blocks of size blockM x blockN. Each block is distributed
    across TMEM 128 rows.

    Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
    convention that needs to be followed by operations reading/writing to TMEM.

    a tensor <128x128xf32> with blockM = 64 and blockN = 32 will be distributed as follows:

        \ col    0        1            31         32            64            96           127
    rows: 0  ( 0,  0) ( 0,  1) ... ( 0,  31)  ( 0,  32) ... ( 0,  64) ... ( 0,  96) ... ( 0,  127)
          1
         ...
          15 (15,  0) (15,  1) ... (15,  31)  (15,  32) ... (15,  64) ... (15,  96) ... (15,  127)
          16 (64,  0) (64,  1) ... (64,  31)  (64,  32) ... (64,  64) ... (64,  96) ... (64,  127)
         ...
          31 (79,  0) (79,  1) ... (79,  31)  (79,  32) ... (79,  64) ... (79,  96) ... (79,  127)
          32 (16,  0) (16,  1) ... (16,  31)  (16,  32) ... (16,  64) ... (16,  96) ... (16,  127)
         ..
         127 (127, 0) (127, 1) ... (127, 31) (127, 32) ... (127, 64) ... (127, 96) ... (127, 127)
  }];
}

def TTNG_TMEMLoadReduceModifierEnum : EnumAttr<TritonNvidiaGPU_Dialect, TTNG_TMEMLoadReduceModifierAttr, "redOp"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNG_TensorMemoryScalesBlockRepOrderEnum
    : EnumAttr<TritonNvidiaGPU_Dialect,
               TTNG_TensorMemoryScalesBlockRepOrderAttr, "blockRepOrder"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNG_CGALayoutRankTwoParam
    : AttrOrTypeParameter<"::mlir::triton::gpu::CGAEncodingAttr",
                          "rank-2 CGA layout"> {
  let parser = [{
    [&]() -> ::mlir::FailureOr<::mlir::triton::gpu::CGAEncodingAttr> {
      return ::mlir::triton::nvidia_gpu::parseCGALayoutRankTwo($_parser);
    }()
  }];
  let printer = [{
    ::mlir::triton::nvidia_gpu::printCGALayoutRankTwo($_printer, $_self);
  }];
  let defaultValue =
      "::mlir::triton::gpu::CGAEncodingAttr::get1CTALayout($_ctxt, 2)";
}

def TTG_TensorMemoryEncodingAttr
    : AttrDef<TritonNvidiaGPU_Dialect,
              "TensorMemoryEncoding", [LayoutEncodingTrait]> {
  let mnemonic = "tensor_memory_encoding";
  let attrName = "triton.gpu.tensor_memory_encoding";
  let description = [{
    An encoding to represent the different way the tensor memory is laid out.
    `colStride` describes the stride in elements along the column dimension,
    that is, the stride between two elements in the same row.
    When colStride is 1 the tensor memory is packed. When colStride > 1, the
    tensor memory between elements is undefined.
    `twoCTAs` indicates that the tensor memory is laid out for twoCTA mode,
    i.e., `cta_group::2`.
  }];
  let parameters = (ins "unsigned":$blockM, "unsigned":$blockN,
      "unsigned":$colStride, TTNG_CGALayoutRankTwoParam:$CGALayout,
      DefaultValuedParameter<"bool", "false">:$twoCTAs,
      DefaultValuedParameter<"TensorMemoryCTAMode",
                             "TensorMemoryCTAMode::DEFAULT">:$ctaMode,
      DefaultValuedParameter<"bool", "false">:$fp4Padded);
  let builders = [AttrBuilder<
      (ins "unsigned":$blockM, "unsigned":$blockN, "unsigned":$colStride,
          "::mlir::triton::gpu::CGAEncodingAttr":$CGALayout, "bool":$twoCTAs,
          "TensorMemoryCTAMode":$ctaMode),
      [{
      return $_get($_ctxt, blockM, blockN, colStride, CGALayout, twoCTAs,
                   ctaMode, false);
    }]>];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

def TTG_TensorMemoryScalesEncodingAttr
    : AttrDef<TritonNvidiaGPU_Dialect,
              "TensorMemoryScalesEncoding", [LayoutEncodingTrait]> {
  let mnemonic = "tensor_memory_scales_encoding";
  let attrName = "triton.gpu.tensor_memory_scales_encoding";
  let description = [{
    An encoding to represent the layout of tensor memory scales.
    As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
    as a multiple copies of a block, each of which having the size 32x4x4B. Moreover, such blocks are duplicated
    over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
    layout.
    The parameter blockRepOrder determines the ordering of repeated scale blocks along TMEM columns. Here `mn`
    denotes the MMA instruction's non-K matrix dimension: `M` for A scales and `N` for B scales. `mnThenK`
    advances along that dimension before advancing K, while `kThenMn` advances along K before advancing the
    non-K dimension. When one MMA instruction consumes only one scale block along K, the value of this parameter
    does not matter. Otherwise, it must match the hardware instruction layout. For example, Rubin NVFP4 A scales
    with `BLOCK_M = 256` require `kThenMn`, while B scales keep the default `mnThenK`.
    `ctaMode = twocta_rhs` represents B scales for the M=128 cta_group::2 Layout-B data path. It moves the high N
    basis from TMEM columns to row partition 64 while keeping row partition 32 broadcast.
  }];
  let parameters =
      (ins TTNG_CGALayoutRankTwoParam:$CGALayout,
          DefaultValuedParameter<
              "::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder",
              "::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder::MN_"
              "THEN_K">:$blockRepOrder,
          DefaultValuedParameter<"TensorMemoryCTAMode",
                                 "TensorMemoryCTAMode::DEFAULT">:$ctaMode);
  let builders = [AttrBuilder<
      (ins "::mlir::triton::gpu::CGAEncodingAttr":$CGALayout), [{
      return get(context, CGALayout,
                 ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder::MN_THEN_K,
                 ::mlir::triton::nvidia_gpu::TensorMemoryCTAMode::DEFAULT);
    }]>];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// BarrierAnnotation attribute
//===----------------------------------------------------------------------===//

def TTNG_BarrierAnnotationAttr
    : AttrDef<TritonNvidiaGPU_Dialect, "BarrierAnnotation"> {
  let mnemonic = "barrier_annotation";
  let description = [{
    Describes where to insert a barrier operation during subtiled region lowering.

    - `barrierIdx`: index into the op's barriers/accumCnts operand lists.
      For tile-region annotations with a tileMask, the lowering computes the
      per-tile barrier index as `(outerAccumCnt + tileIdx) % numBuffers`.
    - `placement`: BEFORE or AFTER the target op
    - `targetOpIdx`: index of the target op in the target region body (0-based,
      counting only non-terminator ops)
    - `barrierOpKind`: "wait_barrier" or "arrive_barrier"
    - `count`: arrive count for arrive_barrier (default 1)
    - `region`: which region the barrier targets (default TILE):
        - TILE: placed in the per-tile body, controlled by tileMask
        - SETUP: placed in the setup region (runs once, no mask)
        - TEARDOWN: placed in the teardown region (runs once, no mask)
    - `numBuffers`: number of buffers for phase and buffer index computation
      (default 1). At lowering time, for each tile replication where
      tileMask[tileIdx] is true:
        tileAccumCnt = outerAccumCnt + tileIdx
        bufferIdx    = tileAccumCnt % numBuffers
        phase        = (tileAccumCnt / numBuffers) & 1
    - `tileMask`: per-tile boolean mask (one entry per tile). The barrier is
      only emitted for tiles where the mask is true. Empty mask means emit
      on all tiles. Only used for TILE region annotations.
  }];
  let parameters = (ins "unsigned":$barrierIdx, "BarrierPlacement":$placement,
      "unsigned":$targetOpIdx, "StringAttr":$barrierOpKind,
      DefaultValuedParameter<"unsigned", "1">:$count,
      DefaultValuedParameter<"BarrierRegion", "BarrierRegion::TILE">:$region,
      DefaultValuedParameter<"unsigned", "1">:$numBuffers,
      OptionalParameter<"DenseI32ArrayAttr">:$tileMask);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TokenAnnotation attribute
//===----------------------------------------------------------------------===//

def TTNG_TokenAnnotationAttr
    : AttrDef<TritonNvidiaGPU_Dialect, "TokenAnnotation"> {
  let mnemonic = "token_annotation";
  let description = [{
    Describes where to insert a token-based synchronization operation during
    subtiled region lowering. This is the token-layer analog of
    `BarrierAnnotationAttr` — it references NVWS tokens (ConsumerWaitOp /
    ConsumerReleaseOp) instead of mbarrier ops (WaitBarrierOp /
    ArriveBarrierOp). Token annotations are resolved to barrier annotations
    during `doTokenLowering`.

    - `tokenIdx`: index into the op's `tokenValues` operand list (the NVWS
      token Value).
    - `bufferIdxIdx`: index into `tokenValues` for the buffer index (i32).
    - `phaseIdx`: index into `tokenValues` for the phase (i1). Set to -1
      for consumer_release ops that have no phase operand.
    - `placement`: BEFORE or AFTER the target op.
    - `targetOpIdx`: index of the target op in the target region body.
    - `tokenOpKind`: "consumer_wait" or "consumer_release".
    - `region`: which region the token op targets (default TILE).
  }];
  let parameters = (ins "unsigned":$tokenIdx, "unsigned":$bufferIdxIdx,
      "int":$phaseIdx, "BarrierPlacement":$placement, "unsigned":$targetOpIdx,
      "StringAttr":$tokenOpKind,
      DefaultValuedParameter<"BarrierRegion", "BarrierRegion::TILE">:$region);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
