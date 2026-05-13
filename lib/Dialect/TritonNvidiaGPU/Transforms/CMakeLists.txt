add_triton_library(TritonNvidiaGPUTransforms
  CheckMatmulTwoCTAs.cpp
  FenceInsertion.cpp
  GenerateSubtiledRegion.cpp
  InterleaveTMem.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  PruneUnusedBarriers.cpp
  ProxyFenceInsertion.cpp
  RemoveTMEMTokens.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp
  TMAStoreBufferReuse.cpp
  TMAUtilities.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
