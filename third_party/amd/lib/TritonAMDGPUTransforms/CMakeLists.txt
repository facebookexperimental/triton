add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  ApplyPlanPipeline.cpp
  PlanRingMaterializer.cpp
  PlanStagingMaterializer.cpp
  ApplyPlanSchedule.cpp
  BlockPingpong.cpp
  CanonicalizePointers.cpp
  CoalesceAsyncCopy.cpp
  CoalesceBufferOps.cpp
  ConSanAMD.cpp
  ConvertToBufferOps.cpp
  ConvertToTensorOps.cpp
  DotDecomposeAndSchedule.cpp
  DumpPlanValueGraph.cpp
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
)

target_link_libraries(TritonAMDGPUTransforms PUBLIC TritonAnalysis)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
