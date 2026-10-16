add_triton_library(NVWSTransforms
  BufferGroups.cpp
  AssignSemaphoreStagePhase.cpp
  OptimizeSemaphore.cpp
  LowerSemaphore.cpp
  LowerAref.cpp
  LowerWarpGroup.cpp
  MetaToNVWSConvert.cpp
  InsertAref.cpp
  Utilities.cpp
  AssignStagePhase.cpp
  InsertTmemAref.cpp
  HoistTmemStore.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
