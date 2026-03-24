add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  TensorMemoryUtils.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen
  TritonNvidiaGPUTypesIncGen
  TLXTableGen
  TLXTypesIncGen
  TLXAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
