#ifndef TRITONNVIDIAGPU_ENUMS
#define TRITONNVIDIAGPU_ENUMS

include "mlir/IR/EnumAttr.td"

def TTNG_CacheEvictionPriorityAttr : I32EnumAttr<
    "CacheEvictionPriority", "NVIDIA cache eviction priority",
    [
      I32EnumAttrCase<"NONE", 0, "none">,
      I32EnumAttrCase<"EVICT_NORMAL", 1, "evict_normal">,
      I32EnumAttrCase<"EVICT_UNCHANGED", 2, "evict_unchanged">,
      I32EnumAttrCase<"EVICT_FIRST", 3, "evict_first">,
      I32EnumAttrCase<"EVICT_LAST", 4, "evict_last">,
      I32EnumAttrCase<"NO_ALLOCATE", 5, "no_allocate">
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

def TTNG_TMEMWaitKindAttr : I32EnumAttr<
    "TMEMWaitKind", "tensor memory wait kind",
    [
        I32EnumAttrCase<"LOAD", 0, "load">,
        I32EnumAttrCase<"STORE", 1, "store">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

def TTNG_TMEMLoadReduceModifierAttr : I32EnumAttr<
    "TMEMLoadReduceModifier", "",
    [
        I32EnumAttrCase<"MIN", 1, "min">,
        I32EnumAttrCase<"MAX", 2, "max">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

def TTNG_PackedArithOpKindAttr : I32EnumAttr<
    "PackedArithOpKind", "packed arithmetic operation",
    [
        I32EnumAttrCase<"ADD", 0, "add">,
        I32EnumAttrCase<"SUB", 1, "sub">,
        I32EnumAttrCase<"MUL", 2, "mul">,
        I32EnumAttrCase<"FMA", 3, "fma">,
        I32EnumAttrCase<"MIN", 4, "min">,
        I32EnumAttrCase<"MAX", 5, "max">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

def TTNG_TensorMemoryScalesBlockRepOrderAttr : I32EnumAttr<
    "TensorMemoryScalesBlockRepOrder", "",
    [
        I32EnumAttrCase<"MN_THEN_K", 0, "mnThenK">,
        I32EnumAttrCase<"K_THEN_MN", 1, "kThenMn">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TensorMemoryCTAMode enum
//===----------------------------------------------------------------------===//

def TTNG_TensorMemoryCTAMode_Default : I32EnumAttrCase<"DEFAULT", 0, "default">;
def TTNG_TensorMemoryCTAMode_TwoCTA_LHS
    : I32EnumAttrCase<"TwoCTA_LHS", 1, "twocta_lhs">;
def TTNG_TensorMemoryCTAMode_TwoCTA_RHS
    : I32EnumAttrCase<"TwoCTA_RHS", 2, "twocta_rhs">;

def TTNG_TensorMemoryCTAMode
    : I32EnumAttr<"TensorMemoryCTAMode",
                  "Tensor memory CTA mode for LinearLayout conversion",
                  [TTNG_TensorMemoryCTAMode_Default,
                   TTNG_TensorMemoryCTAMode_TwoCTA_LHS,
                   TTNG_TensorMemoryCTAMode_TwoCTA_RHS]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

//===----------------------------------------------------------------------===//
// BarrierPlacement enum
//===----------------------------------------------------------------------===//

def TTNG_BarrierPlacementBefore : I32EnumAttrCase<"BEFORE", 0, "before">;
def TTNG_BarrierPlacementAfter : I32EnumAttrCase<"AFTER", 1, "after">;

def TTNG_BarrierPlacement
    : I32EnumAttr<"BarrierPlacement", "Barrier placement relative to target op",
                  [TTNG_BarrierPlacementBefore, TTNG_BarrierPlacementAfter]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

//===----------------------------------------------------------------------===//
// BarrierRegion enum
//===----------------------------------------------------------------------===//

def TTNG_BarrierRegionTile : I32EnumAttrCase<"TILE", 0, "tile">;
def TTNG_BarrierRegionSetup : I32EnumAttrCase<"SETUP", 1, "setup">;
def TTNG_BarrierRegionTeardown : I32EnumAttrCase<"TEARDOWN", 2, "teardown">;

def TTNG_BarrierRegion
    : I32EnumAttr<"BarrierRegion",
                  "Which region of a subtiled_region the barrier targets",
                  [TTNG_BarrierRegionTile, TTNG_BarrierRegionSetup,
                   TTNG_BarrierRegionTeardown]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

def TTNG_TensorModeAttr
    : I32EnumAttr<"TensorMode", "",
                  [I32EnumAttrCase<"TILED", 0, "tiled">,
                   I32EnumAttrCase<"IM2COL", 1, "im2col">]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let description = [{
    Enum attribute for TMA tensor mode.

    TILED: Tiled mode for regular tensor memory access.
    IM2COL: Im2col mode for convolution-friendly tensor memory access.

    See:
    - https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode
    - https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode
  }];
}

#endif // TRITONNVIDIAGPU_ENUMS
