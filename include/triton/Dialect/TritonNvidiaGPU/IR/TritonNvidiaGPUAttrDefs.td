#ifndef TRITONNVIDIAGPU_ATTRDEFS
#define TRITONNVIDIAGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

def TTG_SharedClusterMemorySpace : AttrDef<TritonNvidiaGPU_Dialect, "SharedClusterMemorySpace"> {
  let mnemonic = "shared_cluster_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory. The shared memory could reside in
    any CTA within a CTA cluster.
  }];
}

def TTG_TensorMemorySpace : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemorySpace"> {
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

def TTG_TensorMemoryEncodingAttr : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryEncoding"> {
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
  let parameters = (
    ins
    "unsigned":$blockM,
    "unsigned":$blockN,
    "unsigned":$colStride,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitM,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitN,
    DefaultValuedParameter<"bool", "false">:$twoCTAs
  );
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

def TTG_TensorMemoryScalesEncodingAttr : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryScalesEncoding"> {
  let mnemonic = "tensor_memory_scales_encoding";
  let attrName = "triton.gpu.tensor_memory_scales_encoding";
  let description = [{
    An encoding to represent the layout of tensor memory scales.
    As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
    as a multiple copies of "chunk", each of which having the size 32x4x4B. Moreover, such chunks are duplicated
    over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
    layout.
  }];
  let parameters = (
    ins
    DefaultValuedParameter<"unsigned", "1">:$CTASplitM,
    DefaultValuedParameter<"unsigned", "1">:$CTASplitN
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// BarrierPlacement enum
//===----------------------------------------------------------------------===//

def TTNG_BarrierPlacementBefore    : I32EnumAttrCase<"BEFORE",     0, "before">;
def TTNG_BarrierPlacementAfter     : I32EnumAttrCase<"AFTER",      1, "after">;
def TTNG_BarrierPlacementTileStart : I32EnumAttrCase<"TILE_START", 2, "tile_start">;
def TTNG_BarrierPlacementTileEnd   : I32EnumAttrCase<"TILE_END",   3, "tile_end">;

def TTNG_BarrierPlacement : I32EnumAttr<"BarrierPlacement",
    "Barrier placement relative to tile operations",
    [TTNG_BarrierPlacementBefore, TTNG_BarrierPlacementAfter,
     TTNG_BarrierPlacementTileStart, TTNG_BarrierPlacementTileEnd]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

//===----------------------------------------------------------------------===//
// BarrierRegion enum
//===----------------------------------------------------------------------===//

def TTNG_BarrierRegionTile     : I32EnumAttrCase<"TILE",     0, "tile">;
def TTNG_BarrierRegionSetup    : I32EnumAttrCase<"SETUP",    1, "setup">;
def TTNG_BarrierRegionTeardown : I32EnumAttrCase<"TEARDOWN", 2, "teardown">;

def TTNG_BarrierRegion : I32EnumAttr<"BarrierRegion",
    "Which region of a subtiled_region the barrier targets",
    [TTNG_BarrierRegionTile, TTNG_BarrierRegionSetup,
     TTNG_BarrierRegionTeardown]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

//===----------------------------------------------------------------------===//
// BarrierAnnotation attribute
//===----------------------------------------------------------------------===//

def TTNG_BarrierAnnotationAttr : AttrDef<TritonNvidiaGPU_Dialect, "BarrierAnnotation"> {
  let mnemonic = "barrier_annotation";
  let description = [{
    Describes where to insert a barrier operation during subtiled region lowering.

    - `barrierIdx`: default index into the op's barriers/accumCnts operand lists.
      Used when `bufferIdxArgIdx` is not set (static barrier selection).
    - `placement`: when the barrier fires:
        - BEFORE: before the target op on the first tile only
        - AFTER: after the target op on the last tile only
        - TILE_START: before the target op on every tile
        - TILE_END: after the target op on every tile
    - `targetOpIdx`: index of the target op in the target region body (0-based,
      counting only non-terminator ops)
    - `barrierOpKind`: "wait_barrier" or "arrive_barrier"
    - `count`: arrive count for arrive_barrier (default 1)
    - `region`: which region the barrier targets (default TILE):
        - TILE: placed in the per-tile body
        - SETUP: placed in the setup region (runs once before all tiles)
        - TEARDOWN: placed in the teardown region (runs once after all tiles)
    - `numBuffers`: number of buffers for phase computation (default 1).
      For wait_barrier, the phase is computed at lowering time from the
      accumulation count (accumCnts operand) as:
        phase = (accumCnt / numBuffers) & 1
    - `bufferIdxArgIdx`: optional tile body argument index that provides the
      buffer index per tile (default -1 = not set, use static barrierIdx).
      When set, the lowering uses the tile arg's value (resolved via
      tileMappings) to index into the barriers/accumCnts arrays instead of
      using the static barrierIdx.
  }];
  let parameters = (
    ins
    "unsigned":$barrierIdx,
    "BarrierPlacement":$placement,
    "unsigned":$targetOpIdx,
    "StringAttr":$barrierOpKind,
    DefaultValuedParameter<"unsigned", "1">:$count,
    DefaultValuedParameter<"BarrierRegion", "BarrierRegion::TILE">:$region,
    DefaultValuedParameter<"unsigned", "1">:$numBuffers,
    DefaultValuedParameter<"int", "-1">:$bufferIdxArgIdx
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
