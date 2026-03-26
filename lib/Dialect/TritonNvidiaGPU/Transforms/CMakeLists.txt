add_triton_library(TritonNvidiaGPUTransforms
  FenceInsertion.cpp
  InterleaveTMem.cpp
  LowerSubtiledRegion.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  ProxFenceInsertion.cpp
  RemoveTMEMTokens.cpp
  SubtiledRegionSetupPush.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp
  TMAStoreBufferReuse.cpp
  TMAUtilities.cpp
  Utility.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
