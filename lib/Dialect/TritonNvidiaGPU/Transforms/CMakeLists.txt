add_triton_library(TritonNvidiaGPUTransforms
  CheckMatmulTwoCTAs.cpp
  FenceInsertion.cpp
  GenerateSubtiledRegion.cpp
  InterleaveTMem.cpp
  LowerSubtiledRegion.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PushSharedSetupToTile.cpp
  PromoteLHSToTMem.cpp
  PruneUnusedBarriers.cpp
  ProxFenceInsertion.cpp
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
