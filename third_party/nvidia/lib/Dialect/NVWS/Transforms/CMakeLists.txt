add_triton_library(NVWSTransforms
  BufferGroups.cpp
  AssignSemaphoreStagePhase.cpp
  OptimizeSemaphore.cpp
  LowerSemaphore.cpp
  LowerAref.cpp
  LowerWarpGroup.cpp
  MetaToNVWSConvert.cpp
  PackEpilogueSlices.cpp
  OrderBufferGroups.cpp
  InsertAref.cpp
  InsertAllocas.cpp
  InsertSemas.cpp
  InsertSemasAccessDag.cpp
  InsertSemasEmitIR.cpp
  InsertSemasSyncDag.cpp
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
