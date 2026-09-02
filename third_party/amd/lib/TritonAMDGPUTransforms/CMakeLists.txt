add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  BlockPingpong.cpp
  CanonicalizePointers.cpp
  CoalesceAsyncCopy.cpp
  CoalesceBufferOps.cpp
  ConSanAMD.cpp
  ConvertToBufferOps.cpp
  ConvertToTensorOps.cpp
  DotDecomposeAndSchedule.cpp
  SchedGroupBarrierScheduler.cpp
  LowerBarrierOps.cpp
  OptimizeBufferOpPtr.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeEpilogue.cpp
  OptimizeDotOperands.cpp
  HoistLayoutConversions.cpp
  SinkLayoutConversions.cpp
  PrepareIfCombining.cpp
  MoveUpPrologueLoads.cpp
  Pipeline.cpp
  ScheduleLoops.cpp
  LowerLoops.cpp
  TDMStoresPipeline.cpp
  MfmaGroup.cpp
  WmmaGroup.cpp
  InThreadTranspose.cpp
  FoldTrueCmpIOp.cpp
  FpSanitizer.cpp
  UpdateAsyncWaitCount.cpp
  Utility.cpp
  WarpPipeliner.cpp

  DEPENDS
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
  TritonGPUModuloCore
  TritonAMDUtils
  TritonAMDAnalysis
  TritonInstrumentTransforms
  TritonInstrumentIR

  LINK_LIBS PUBLIC
  TritonAMDUtils
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
