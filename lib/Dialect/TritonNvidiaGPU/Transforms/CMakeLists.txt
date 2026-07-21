add_triton_library(TritonNvidiaGPUTransforms
  ClusterBarrierInsertion.cpp
  CheckMatmulTwoCTAs.cpp
  CLCLowering.cpp
  ConSanNVIDIA.cpp
  FenceInsertion.cpp
  FuseTMemLoadReduce.cpp
  GenerateSubtiledRegion.cpp
  InterleaveTMem.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  PromoteLoadToTMA.cpp
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
  TritonInstrumentTransforms
  TritonInstrumentIR
  MLIRTransformUtils
)
