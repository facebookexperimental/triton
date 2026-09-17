add_triton_library(NVWSTransforms
  BufferGroups.cpp
  AssignSemaphoreStagePhase.cpp
  OptimizeSemaphore.cpp
  LowerSemaphore.cpp
  LowerWarpGroup.cpp
  InsertAllocas.cpp
  MetaToNVWSConvert.cpp
  OrderBufferGroups.cpp
  PackEpilogueSlices.cpp
  Utilities.cpp
  InsertSemas.cpp
  InsertSemasAccessDag.cpp
  InsertSemasEmitIR.cpp
  InsertSemasSyncDag.cpp
  HoistTmemStore.cpp
  StripPartitionAttrsOutsideWS.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVHopperTransforms
  NVWSIR
  MLIRTransformUtils
)
