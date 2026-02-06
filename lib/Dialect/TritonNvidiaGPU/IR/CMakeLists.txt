add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen
  TLXTableGen
  TLXTypesIncGen
  TLXAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
