add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  Membar.cpp
  Alias.cpp
  PlanAsyncLifetime.cpp
  PlanAudit.cpp
  PlanSchedule.cpp
  PlanValueGraph.cpp
  PlanLiveness.cpp
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
  MLIRLLVMDialect
  MLIRSCFDialect
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
  TritonInstrumentIR
)
