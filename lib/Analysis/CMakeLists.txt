add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  BufferIndexAnalysis.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp
  ReductionOrder.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
  TritonInstrumentTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
  TritonInstrumentIR
)
