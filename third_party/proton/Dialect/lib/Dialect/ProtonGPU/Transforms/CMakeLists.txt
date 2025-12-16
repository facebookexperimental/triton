add_triton_library(ProtonGPUTransforms
  ProtonGPUTransformsPass.cpp
  MppStoreBarrierInfoPass.cpp

  DEPENDS
  ProtonGPUTransformsIncGen
  LINK_LIBS PUBLIC
  ProtonGPUIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRSCFDialect
  MLIRArithDialect
)
