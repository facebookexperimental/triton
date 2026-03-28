add_triton_library(TritonNvidiaGPUTransforms
  BarrierAnalysisPass.cpp
  FenceInsertion.cpp
  InterleaveTMem.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  ProxFenceInsertion.cpp
  RemoveTMEMTokens.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp
  TMAStoreBufferReuse.cpp
  TMAUtilities.cpp
  Utility.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
